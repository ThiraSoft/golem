"""Records what the THA3 separable_float poser computes, for the Go tests.

Usage (from the venv of ref/tha3/README.md):
    python ref/tha3/dump.py <demo checkout> <dir with the .pt files> <picture> <out dir>

Runs the upstream code, unmodified, on the CPU in float32. For every pose
below it writes <out dir>/<pose>/<name>.bin (raw float32) and fixtures.json.
Forward wrappers catch what goes in and out of each network; for the first pose
hooks also catch the output of every block inside each body.
"""

import json
import sys
from pathlib import Path

import torch

POSES = {
    "neutral": {},
    "aaa": {26: 1.0},
    "wink": {12: 1.0},
    "head": {39: 0.8},
    "mix": {2: 0.6, 3: 0.6, 15: 0.5, 30: 0.7, 37: 0.5, 40: -0.5, 41: 0.3,
            42: 0.4, 43: -0.3, 44: 0.5},
}
NETWORKS = ["eyebrow_decomposer", "eyebrow_morphing_combiner", "face_morpher",
            "two_algo_face_body_rotator", "editor"]
BODIES = {"two_algo_face_body_rotator": "encoder_decoder"}


def main() -> None:
    demo, weights, picture, out = map(Path, sys.argv[1:5])
    sys.path.insert(0, str(demo))
    from tha3.poser.modes.separable_float import create_poser, FiveStepPoserComputationProtocol
    from tha3.nn.eyebrow_morphing_combiner.eyebrow_morphing_combiner_00 import EyebrowMorphingCombiner00
    from tha3.util import extract_pytorch_image_from_filelike

    files = {n: str(weights / f"{n}.pt") for n in NETWORKS}
    poser = create_poser(torch.device("cpu"), module_file_names=files)
    modules = poser.get_modules()
    image = extract_pytorch_image_from_filelike(str(picture)).unsqueeze(0)

    for index, (run, values) in enumerate(POSES.items()):
        directory = out / run
        directory.mkdir(parents=True, exist_ok=True)
        shapes: dict[str, list[int]] = {}

        def dump(name: str, tensor: torch.Tensor) -> None:
            array = tensor.detach().contiguous().float()
            (directory / f"{name}.bin").write_bytes(array.numpy().tobytes())
            shapes[name] = list(array.shape)

        orig_forwards = {}
        for net in NETWORKS:
            module = modules[net]
            orig_forwards[net] = module.forward

            def make_wrapper(net_name, orig_forward):
                def wrapped_forward(*args, **kwargs):
                    for i, a in enumerate(args):
                        if isinstance(a, torch.Tensor):
                            dump(f"{net_name}.in.{i}", a)
                    outputs = orig_forward(*args, **kwargs)
                    if isinstance(outputs, (list, tuple)):
                        for i, o in enumerate(outputs):
                            if isinstance(o, torch.Tensor):
                                dump(f"{net_name}.out.{i}", o)
                    elif isinstance(outputs, torch.Tensor):
                        dump(f"{net_name}.out.0", outputs)
                    return outputs
                return wrapped_forward

            module.forward = make_wrapper(net, orig_forwards[net])

        handles = []
        if index == 0:
            for net in NETWORKS:
                module = modules[net]
                body = BODIES.get(net, "body")
                for part in ("downsample_blocks", "bottleneck_blocks", "upsample_blocks"):
                    for i, block in enumerate(getattr(getattr(module, body), part)):
                        def one(_, __, output, name=f"{net}.{body}.{part}.{i}"):
                            dump(name, output)
                        handles.append(block.register_forward_hook(one))

        pose = torch.zeros(1, 45)
        for i, v in values.items():
            pose[0, i] = v
        # A fresh protocol per run: its cache of the decomposer's output would
        # otherwise skip the decomposer, and its hook with it, after the first
        # pose.
        poser.output_list_func = FiveStepPoserComputationProtocol(
            EyebrowMorphingCombiner00.EYEBROW_IMAGE_NO_COMBINE_ALPHA_INDEX).compute_func()
        dump("image", image)
        with torch.no_grad():
            poser.get_posing_outputs(image, pose)
        for h in handles:
            h.remove()
        for net in NETWORKS:
            modules[net].forward = orig_forwards[net]
        fixtures = {"pose": pose[0].tolist(), "tensors": shapes}
        (directory / "fixtures.json").write_text(json.dumps(fixtures, indent=1, sort_keys=True))
        print(f"{run}: {len(shapes)} tensors")


if __name__ == "__main__":
    main()
