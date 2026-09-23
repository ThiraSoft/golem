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
err = p.UseVulkan() // optional: SetImage and Pose on the card, or an error
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

`SetScale(k)` and `SetImageHigh(img, high)` make the frame k times larger
without running the networks any larger: what they decide at 512 (where each
pixel goes, where they repaint) is resized and applied to `high`, the same
picture at 512·k, so that what they only move keeps its detail. At k = 4 on
the card, a pose costs about 1.5 ms more. `SetBackground` and `PoseRGBA`
finish the frame on the card (laid on a colour, sRGB, framed by a zoom) so
that only bytes come back.

`SetSharpen(a)` brings back the edge of what the face morpher paints, which
it decides at 192 px and the larger picture blows up. `SetEyeTone(a)` deals
with the colour of the same paint: the networks give a closed eyelid the skin
they were trained on, warmer than most characters', which at 512 passes and
on the larger picture reads as a beige patch on a face the picture drew. At
one the gain measured on the picture itself brings it onto the character's
own skin, at the cost of one run of the face morpher per picture set and
nothing per frame.

`SetEyeUpscale(a)` goes further than the sharpening for the eyes: the face
the morpher made at 192 goes through an anime upscaler, four times larger,
and is laid over the larger picture's face where the eyes were repainted,
fading out around them. A closed eye's lash line comes back as a line. It
needs the upscaler's weights, which `ref/tha3/README.md` says how to fetch.
At a scale of two on the card it costs about 6 ms a pose: 12.0 ms without,
18.0 ms with, on the moving sequence of `cmd/tha3-pose -bench`.

`SetFront(mask)` says which pixels of the picture stay in front of the face.
The morpher repaints the whole eye socket when an eye shuts, over whatever
happens to be there, so a lock of hair falling across an eye flickers with
every blink; the mask puts it back, before the rotator, so that it still
turns with the head. It is used as it is given. `SetFrontHigh(mask, high)`
adds the larger picture's own mask: a lock one pixel wide at 512, blown up
from a mask of that size, comes back as a smudge.

`cmd/tha3-pose` renders one frame from the command line and prints how long
each network took.

## Measured

Measured on an Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz using 8 threads
(nproc 8). The table is one pose after SetImage.

| Network | Time (ms) |
| --- | ---: |
| editor | 1140.8 |
| eyebrow_morphing_combiner | 122.6 |
| face_morpher | 285.3 |
| two_algo_face_body_rotator | 628.6 |
| total | 2179.6 |
| eyebrow_decomposer (once-per-picture cost of SetImage) | 306.9 |

### On the card

Measured on an AMD Radeon RX 9070 (RADV driver, float32, arena 298 MB).
One pose, wall clock around `Pose` after `SetImage`: 19.9 ms. Sustained,
300 poses of a moving sequence: 54.7 poses/s, 18.29 ms each. A pose equal
to the last one returns the last frame without computing. The 33 ms target
is met with comfortable margin, and the editor is the slowest network.

| Network | Time (ms) |
| --- | ---: |
| editor | 8.1 |
| eyebrow_morphing_combiner | 2.3 |
| face_morpher | 3.4 |
| two_algo_face_body_rotator | 3.7 |
| pose (without decomposer) | 19.9 |
| eyebrow_decomposer (once-per-picture cost of SetImage) | 3.8 |

