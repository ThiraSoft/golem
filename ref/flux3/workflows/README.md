# ComfyUI workflows

FLUX 3 Image Klein workflows with the prompt upsampler built in. Each takes a prompt and optional bounding boxes; the image workflows also take reference images.

| Checkpoint | Text to image | Image to image | Multiple images to image |
| --- | --- | --- | --- |
| Distilled, BF16 | [flux3_klein_bf16_t2i](./flux3_klein_bf16_t2i.json) | [flux3_klein_bf16_i2i](./flux3_klein_bf16_i2i.json) | [flux3_klein_bf16_mi2i](./flux3_klein_bf16_mi2i.json) |
| Distilled, int8 | [flux3_klein_int8_t2i](./flux3_klein_int8_t2i.json) | [flux3_klein_int8_i2i](./flux3_klein_int8_i2i.json) | [flux3_klein_int8_mi2i](./flux3_klein_int8_mi2i.json) |
| Base, BF16 | [flux3_klein_base_bf16_t2i](./flux3_klein_base_bf16_t2i.json) | [flux3_klein_base_bf16_i2i](./flux3_klein_base_bf16_i2i.json) | [flux3_klein_base_bf16_mi2i](./flux3_klein_base_bf16_mi2i.json) |
| Base, int8 | [flux3_klein_base_int8_t2i](./flux3_klein_base_int8_t2i.json) | [flux3_klein_base_int8_i2i](./flux3_klein_base_int8_i2i.json) | [flux3_klein_base_int8_mi2i](./flux3_klein_base_int8_mi2i.json) |

## Setup

1. Update ComfyUI to the latest version.
2. Download the models listed in each workflow's note into `ComfyUI/models/`.
3. Copy the sample images (`klein_ref_*.jpg`) into `ComfyUI/input/`.
4. Drag a workflow onto the canvas.

Bounding boxes need Nodes 2.0 (ComfyUI icon → Nodes 2.0).

Workflow files: MIT, see [LICENSE.upstream](./LICENSE.upstream).
