// Command tha3-pose renders one frame of a THA3 character: a picture, a pose
// given as name=value pairs, a PNG out, and how long each network took.
//
//	tha3-pose -image avatar_tha.png -out frame.png mouth_aaa=1 head_x=0.5
package main

import (
	"flag"
	"fmt"
	"image/png"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/ThiraSoft/golem/tha3"
)

var names = map[string]int{
	"eyebrow_troubled_left": tha3.EyebrowTroubledLeft, "eyebrow_troubled_right": tha3.EyebrowTroubledRight,
	"eyebrow_angry_left": tha3.EyebrowAngryLeft, "eyebrow_angry_right": tha3.EyebrowAngryRight,
	"eyebrow_lowered_left": tha3.EyebrowLoweredLeft, "eyebrow_lowered_right": tha3.EyebrowLoweredRight,
	"eyebrow_raised_left": tha3.EyebrowRaisedLeft, "eyebrow_raised_right": tha3.EyebrowRaisedRight,
	"eyebrow_happy_left": tha3.EyebrowHappyLeft, "eyebrow_happy_right": tha3.EyebrowHappyRight,
	"eyebrow_serious_left": tha3.EyebrowSeriousLeft, "eyebrow_serious_right": tha3.EyebrowSeriousRight,
	"eye_wink_left": tha3.EyeWinkLeft, "eye_wink_right": tha3.EyeWinkRight,
	"eye_happy_wink_left": tha3.EyeHappyWinkLeft, "eye_happy_wink_right": tha3.EyeHappyWinkRight,
	"eye_surprised_left": tha3.EyeSurprisedLeft, "eye_surprised_right": tha3.EyeSurprisedRight,
	"eye_relaxed_left": tha3.EyeRelaxedLeft, "eye_relaxed_right": tha3.EyeRelaxedRight,
	"eye_unimpressed_left": tha3.EyeUnimpressedLeft, "eye_unimpressed_right": tha3.EyeUnimpressedRight,
	"eye_raised_lower_eyelid_left": tha3.EyeRaisedLowerEyelidLeft, "eye_raised_lower_eyelid_right": tha3.EyeRaisedLowerEyelidRight,
	"iris_small_left": tha3.IrisSmallLeft, "iris_small_right": tha3.IrisSmallRight,
	"mouth_aaa": tha3.MouthAaa, "mouth_iii": tha3.MouthIii, "mouth_uuu": tha3.MouthUuu,
	"mouth_eee": tha3.MouthEee, "mouth_ooo": tha3.MouthOoo, "mouth_delta": tha3.MouthDelta,
	"mouth_lowered_corner_left": tha3.MouthLoweredCornerLeft, "mouth_lowered_corner_right": tha3.MouthLoweredCornerRight,
	"mouth_raised_corner_left": tha3.MouthRaisedCornerLeft, "mouth_raised_corner_right": tha3.MouthRaisedCornerRight,
	"mouth_smirk":     tha3.MouthSmirk,
	"iris_rotation_x": tha3.IrisRotationX, "iris_rotation_y": tha3.IrisRotationY,
	"head_x": tha3.HeadX, "head_y": tha3.HeadY, "neck_z": tha3.NeckZ,
	"body_y": tha3.BodyY, "body_z": tha3.BodyZ, "breathing": tha3.Breathing,
}

func main() {
	picture := flag.String("image", "", "a 512x512 RGBA PNG following the THA3 template")
	out := flag.String("out", "frame.png", "where to write the frame")
	dir := flag.String("weights", tha3.Dir(), "directory of the converted weights")
	repeat := flag.Int("n", 1, "render the pose this many times and report the last")
	vulkan := flag.Bool("vulkan", false, "run on the first Vulkan device")
	bench := flag.Int("bench", 0, "render this many poses of a moving sequence and print poses per second")
	flag.Parse()
	if *picture == "" {
		fail(fmt.Errorf("-image is required"))
	}

	var pose [tha3.NumParams]float32
	for _, arg := range flag.Args() {
		name, value, ok := strings.Cut(arg, "=")
		index, known := names[name]
		if !ok || !known {
			fail(fmt.Errorf("%q is not name=value with a known name", arg))
		}
		v, err := strconv.ParseFloat(value, 32)
		if err != nil {
			fail(fmt.Errorf("%s: %w", name, err))
		}
		pose[index] = float32(v)
	}

	p, err := tha3.Open(*dir)
	if err != nil {
		fail(err)
	}
	defer p.Close()
	if *vulkan {
		if err := p.UseVulkan(); err != nil {
			fail(err)
		}
	}
	img, err := tha3.LoadImage(*picture)
	if err != nil {
		fail(err)
	}
	if err := p.SetImage(img); err != nil {
		fail(err)
	}
	if *bench > 0 {
		var seq [tha3.NumParams]float32
		start := time.Now()
		for i := 0; i < *bench; i++ {
			phase := float64(i) / 15
			seq[tha3.MouthAaa] = float32(0.5 + 0.5*math.Sin(phase*3))
			seq[tha3.HeadY] = float32(0.5 * math.Sin(phase))
			seq[tha3.Breathing] = float32(0.5 + 0.5*math.Sin(phase/2))
			if _, err := p.Pose(seq); err != nil {
				fail(err)
			}
		}
		elapsed := time.Since(start)
		fmt.Printf("%d poses in %v: %.1f poses/s, %.2f ms each\n", *bench, elapsed.Round(time.Millisecond),
			float64(*bench)/elapsed.Seconds(), float64(elapsed.Microseconds())/1000/float64(*bench))
	}
	var frame tha3.Tensor
	var total time.Duration
	for i := 0; i < *repeat; i++ {
		start := time.Now()
		if frame, err = p.Pose(pose); err != nil {
			fail(err)
		}
		total = time.Since(start)
	}

	f, err := os.Create(*out)
	if err != nil {
		fail(err)
	}
	if err := png.Encode(f, tha3.ToNRGBA(frame)); err != nil {
		fail(err)
	}
	if err := f.Close(); err != nil {
		fail(err)
	}

	timings := p.Timings()
	keys := make([]string, 0, len(timings))
	for k := range timings {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Printf("%-28s %8.1f ms\n", k, float64(timings[k].Microseconds())/1000)
	}
	fmt.Printf("%-28s %8.1f ms\n", "pose (without decomposer)", float64(total.Microseconds())/1000)
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "tha3-pose:", err)
	os.Exit(1)
}
