# tha3

Talking Head(?) Anime from a Single Image 3, by Pramook Khungurn: one picture
of an anime character and 45 numbers in, the character with that expression
and head pose out. This is the demo's `separable_float` variant, transcribed
into Go and checked against the demo's PyTorch network by network.

The weights are CC-BY 4.0 and are not in this repository.
`ref/tha3/README.md` says how to fetch and convert them, and how to record
the fixtures the tests compare against.

```go
p, err := tha3.Open(tha3.Dir())
img, err := tha3.LoadImage("character.png")
err = p.SetImage(img)
var pose [tha3.NumParams]float32
pose[tha3.MouthAaa] = 1
frame, err := p.Pose(pose)
out := tha3.ToNRGBA(frame)
```

The picture must be 512×512 with an alpha channel, the background at alpha 0,
the character upright and facing forward, the head inside the 128×128 box
centred in the top half, the hands away from the head.

`cmd/tha3-pose` renders one frame from the command line and prints how long
each network took.

## Measured

Measured on an Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz using 8 threads (nproc 8) in pure Go CPU mode. The numbers record the execution time of one pose after SetImage has cached the decomposed background. Note that the eyebrow decomposer runs once per picture in SetImage, not per pose.

| Network | Time (ms) |
| --- | ---: |
| editor | 1140.8 |
| eyebrow_decomposer | 306.9 |
| eyebrow_morphing_combiner | 122.6 |
| face_morpher | 285.3 |
| two_algo_face_body_rotator | 628.6 |
| pose (without decomposer) | 2179.6 |
