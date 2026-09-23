package qwen35

// Prism ML activation rotation and permutation binding.
//
// Models quantized with Prism ML's Bonsai pipeline apply a Hadamard transform
// and sign flips to activations before multiplying against ternary weights.
// The metadata in prism.hadamard.* specifies the transform parameters, sign
// vectors per layer width, and weight names to rotate.
//
// For SSM blocks with grouped GDN value heads (prism.hadamard.gdn_v_grouped),
// the SSMOut activation is permuted from tiled head order into grouped head
// order prior to the sign flips and rotation.

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ThiraSoft/golem/nn"
	"github.com/ThiraSoft/golem/tensors"
)

// bindPrism applies Prism's activation rotation and permutation to the weights.
// If the file does not carry Prism rotation metadata, it does nothing and returns nil.
func bindPrism(g *tensors.GGUF, w *Weights, cfg *Config) error {
	if _, ok := g.Meta["prism.hadamard.version"]; !ok {
		return nil
	}

	version, err := g.Uint32("prism.hadamard.version")
	if err != nil || version != 1 {
		return fmt.Errorf("unsupported prism.hadamard.version: %v (want 1)", version)
	}

	transform, err := g.String("prism.hadamard.transform")
	if err != nil || transform != "normalized-sylvester-walsh-hadamard" {
		return fmt.Errorf("unsupported prism.hadamard.transform: %q (want normalized-sylvester-walsh-hadamard)", transform)
	}

	axis, err := g.String("prism.hadamard.axis")
	if err != nil || axis != "input-last-dimension" {
		return fmt.Errorf("unsupported prism.hadamard.axis: %q (want input-last-dimension)", axis)
	}

	signMode, err := g.String("prism.hadamard.sign_mode")
	if err != nil || (signMode != "explicit" && signMode != "identity") {
		return fmt.Errorf("unsupported prism.hadamard.sign_mode: %q (want explicit or identity)", signMode)
	}

	blockSize, err := g.Uint32("prism.hadamard.block_size")
	if err != nil || blockSize == 0 || (blockSize&(blockSize-1)) != 0 {
		return fmt.Errorf("unsupported prism.hadamard.block_size: %d (must be a power of two)", blockSize)
	}

	signs := make(map[int][]float32)
	switch signMode {
	case "explicit":
		signWidths, err := g.Uint32Slice("prism.hadamard.sign_widths")
		if err != nil {
			return fmt.Errorf("reading prism.hadamard.sign_widths: %w", err)
		}
		signValues, err := g.Int32Slice("prism.hadamard.sign_values")
		if err != nil {
			return fmt.Errorf("reading prism.hadamard.sign_values: %w", err)
		}
		totalWidth := 0
		for _, width := range signWidths {
			totalWidth += int(width)
		}
		if len(signValues) != totalWidth {
			return fmt.Errorf("prism.hadamard.sign_values length %d does not match sum of sign_widths %d", len(signValues), totalWidth)
		}
		offset := 0
		for _, width := range signWidths {
			wInt := int(width)
			vec := make([]float32, wInt)
			for i := 0; i < wInt; i++ {
				val := signValues[offset+i]
				if val != 1 && val != -1 {
					return fmt.Errorf("invalid sign value %d at index %d (must be +1 or -1)", val, offset+i)
				}
				vec[i] = float32(val)
			}
			signs[wInt] = vec
			offset += wInt
		}
	case "identity":
		if signWidths, err := g.Uint32Slice("prism.hadamard.sign_widths"); err == nil {
			for _, width := range signWidths {
				wInt := int(width)
				vec := make([]float32, wInt)
				for i := range vec {
					vec[i] = 1.0
				}
				signs[wInt] = vec
			}
		}
	}

	getSignVector := func(cols int) ([]float32, bool) {
		if s, ok := signs[cols]; ok && len(s) == cols {
			return s, true
		}
		if signMode == "identity" {
			vec := make([]float32, cols)
			for i := range vec {
				vec[i] = 1.0
			}
			signs[cols] = vec
			return vec, true
		}
		return nil, false
	}

	weightNames, err := g.Strings("prism.hadamard.weight_names")
	if err != nil {
		return fmt.Errorf("reading prism.hadamard.weight_names: %w", err)
	}
	for _, name := range weightNames {
		m, err := findPrismMatrix(w, name)
		if err != nil {
			return err
		}
		s, ok := getSignVector(m.Cols)
		if !ok {
			return fmt.Errorf("matrix %s has Cols=%d with no sign vector", name, m.Cols)
		}
		m.Pre = s
		m.HadGroup = int(blockSize)
	}

	invWeightNames, err := g.Strings("prism.hadamard.inverse_weight_names")
	if err != nil {
		return fmt.Errorf("reading prism.hadamard.inverse_weight_names: %w", err)
	}
	for _, name := range invWeightNames {
		if name != "token_embd.weight" {
			return fmt.Errorf("unsupported inverse_weight_name: %q (only token_embd.weight allowed)", name)
		}
		s, ok := getSignVector(cfg.Dim)
		if !ok {
			return fmt.Errorf("token_embd.weight has Dim=%d with no sign vector", cfg.Dim)
		}
		w.TokenEmbd.Pre = s
		w.TokenEmbd.HadGroup = int(blockSize)
	}

	if gdnVGrouped, err := g.Bool("prism.hadamard.gdn_v_grouped"); err == nil && gdnVGrouped {
		for i := range cfg.Blocks {
			bc := &cfg.Blocks[i]
			if bc.Type != BlockSSM {
				continue
			}
			Inner := bc.SSMInnerSize
			Rank := bc.SSMTimeStepRank
			nk := bc.SSMGroupCount
			if Rank == 0 || Inner%Rank != 0 {
				return fmt.Errorf("SSMInnerSize %d not divisible by SSMTimeStepRank %d", Inner, Rank)
			}
			if nk == 0 || Rank%nk != 0 {
				return fmt.Errorf("SSMTimeStepRank %d not divisible by SSMGroupCount %d", Rank, nk)
			}
			hd := Inner / Rank
			rep := Rank / nk
			gather := make([]int32, Inner)
			for k := 0; k < nk; k++ {
				for r := 0; r < rep; r++ {
					for d := 0; d < hd; d++ {
						gather[d+hd*(r+rep*k)] = int32(d + hd*(k+nk*r))
					}
				}
			}
			w.Blocks[i].SSMOutGather = gather
		}
	}

	checkTernaryRotated := func(name string, m *nn.Matrix) error {
		if (m.Quant == nn.PQ2_0 || m.Quant == nn.PTQ1_0) && m.Pre == nil {
			return fmt.Errorf("matrix %s is %s but has no Pre vector (unrotated)", name, m.Quant)
		}
		return nil
	}

	if err := checkTernaryRotated("token_embd.weight", &w.TokenEmbd); err != nil {
		return err
	}
	if err := checkTernaryRotated("output.weight", &w.OutputHead); err != nil {
		return err
	}
	for i := range w.Blocks {
		bw := &w.Blocks[i]
		prefix := fmt.Sprintf("blk.%d.", i)
		if cfg.Blocks[i].Type == BlockFullAttn {
			if err := checkTernaryRotated(prefix+"attn_q.weight", &bw.Q); err != nil {
				return err
			}
			if err := checkTernaryRotated(prefix+"attn_k.weight", &bw.K); err != nil {
				return err
			}
			if err := checkTernaryRotated(prefix+"attn_v.weight", &bw.V); err != nil {
				return err
			}
			if err := checkTernaryRotated(prefix+"attn_output.weight", &bw.O); err != nil {
				return err
			}
		} else {
			if err := checkTernaryRotated(prefix+"attn_qkv.weight", &bw.QKV); err != nil {
				return err
			}
			if err := checkTernaryRotated(prefix+"attn_gate.weight", &bw.AttnGate); err != nil {
				return err
			}
			if err := checkTernaryRotated(prefix+"ssm_out.weight", &bw.SSMOut); err != nil {
				return err
			}
			if err := checkTernaryRotated(prefix+"ssm_alpha.weight", &bw.SSMAlpha); err != nil {
				return err
			}
			if err := checkTernaryRotated(prefix+"ssm_beta.weight", &bw.SSMBeta); err != nil {
				return err
			}
		}
		if err := checkTernaryRotated(prefix+"ffn_gate.weight", &bw.Gate); err != nil {
			return err
		}
		if err := checkTernaryRotated(prefix+"ffn_up.weight", &bw.Up); err != nil {
			return err
		}
		if err := checkTernaryRotated(prefix+"ffn_down.weight", &bw.Down); err != nil {
			return err
		}
	}

	w.Rotated = true
	return nil
}

// findPrismMatrix maps a weight name from prism.hadamard.weight_names to its nn.Matrix in Weights.
func findPrismMatrix(w *Weights, name string) (*nn.Matrix, error) {
	if name == "output.weight" {
		return &w.OutputHead, nil
	}
	if strings.HasPrefix(name, "blk.") && strings.HasSuffix(name, ".weight") {
		trimmed := strings.TrimPrefix(name, "blk.")
		trimmed = strings.TrimSuffix(trimmed, ".weight")
		dot := strings.IndexByte(trimmed, '.')
		if dot <= 0 {
			return nil, fmt.Errorf("cannot map weight %q", name)
		}
		blockIdx, err := strconv.Atoi(trimmed[:dot])
		if err != nil || blockIdx < 0 || blockIdx >= len(w.Blocks) {
			return nil, fmt.Errorf("cannot map weight %q: block index out of range", name)
		}
		kind := trimmed[dot+1:]
		bw := &w.Blocks[blockIdx]
		switch kind {
		case "attn_q":
			return &bw.Q, nil
		case "attn_k":
			return &bw.K, nil
		case "attn_v":
			return &bw.V, nil
		case "attn_output":
			return &bw.O, nil
		case "attn_qkv":
			return &bw.QKV, nil
		case "attn_gate":
			return &bw.AttnGate, nil
		case "ssm_out":
			return &bw.SSMOut, nil
		case "ffn_gate":
			return &bw.Gate, nil
		case "ffn_up":
			return &bw.Up, nil
		case "ffn_down":
			return &bw.Down, nil
		default:
			return nil, fmt.Errorf("cannot map weight %q: unknown kind %q", name, kind)
		}
	}
	return nil, fmt.Errorf("cannot map weight %q", name)
}
