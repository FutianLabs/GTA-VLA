#!/usr/bin/env python3
"""
CoT Pretrain Verification Demo

Web demo for verifying CoT (Chain-of-Thought) pretraining quality.
Loads a pretrained XVLA checkpoint and generates CoT predictions given
an image + instruction, then visualizes the results.

Supports interactive annotations: draw bounding boxes, click pick points,
and draw gripper trajectories on the image. Annotations are injected into
the instruction using the same format as InteractionAugmenter in training.

Usage:
    python tools/cot_demo.py \
        --checkpoint /path/to/ckpt \
        --config configs/pretrain/gtavla_qwen3vl_2b_cot_pretrain.json \
        --port 7860

    # With second view image:
    python tools/cot_demo.py \
        --checkpoint /path/to/ckpt --config /path/to/config.json \
        --port 7860
"""

import argparse
import sys
import os
import time
from typing import Dict, List, Tuple

# Ensure localhost is excluded from proxy so Gradio's health check works
_no_proxy = os.environ.get("no_proxy", "")
if "localhost" not in _no_proxy:
    os.environ["no_proxy"] = f"{_no_proxy},localhost,127.0.0.1" if _no_proxy else "localhost,127.0.0.1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr

from models.configuration_gtavla import XVLAConfig
from models.modeling_gtavla import XVLA
from models.processing_gtavla import build_xvla_processor
from models.vla_factory import _load_checkpoint_keep_mismatch
from scripts.visualize_cot import parse_cot_elements, visualize_cot_on_image

# ---------------------------------------------------------------------------
# Annotation constants (colors matching visualize_cot.py)
# ---------------------------------------------------------------------------
COLOR_PICK_BOX = (0, 255, 0)        # green
COLOR_AFFORDANCE = (255, 255, 0)    # yellow
COLOR_PATH_START = (0, 255, 0)      # green
COLOR_PATH_END = (255, 0, 0)        # red

IMG_SIZE = (256, 256)


def _empty_annotations() -> Dict:
    return {
        "pick_box": None,
        "pick_point": None,
        "gripper_path": [],
        "click_buffer": [],
    }


# ---------------------------------------------------------------------------
# Interactive annotation helpers
# ---------------------------------------------------------------------------

def pixel_to_model(x: int, y: int, img_w: int = 256, img_h: int = 256, scale: int = 1000) -> Tuple[int, int]:
    """Convert pixel coordinates to model coordinates (0-scale)."""
    return round(x * scale / img_w), round(y * scale / img_h)


def resample_path(points: List[Tuple[int, int]], n: int = 5) -> List[Tuple[int, int]]:
    """Resample a polyline to *n* equidistant points (cumulative-distance based).

    Same algorithm as ``datasets.cot_builder.sample_gripper_path``.
    """
    if not points:
        return []
    if len(points) == 1:
        return [points[0]] * n
    pts = np.array(points, dtype=float)
    cum_dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if cum_dist[-1] == 0:
        return [points[0]] * n
    targets = np.linspace(0, cum_dist[-1], n)
    sx = np.interp(targets, cum_dist, pts[:, 0])
    sy = np.interp(targets, cum_dist, pts[:, 1])
    return [(int(x), int(y)) for x, y in zip(sx, sy)]


def draw_annotations(image: np.ndarray, ann: Dict) -> np.ndarray:
    """Draw current user annotations on the image and return a new array."""
    pil = Image.fromarray(image.copy()).convert("RGB")
    draw = ImageDraw.Draw(pil)

    if ann.get("pick_box"):
        (x1, y1), (x2, y2) = ann["pick_box"]
        draw.rectangle([x1, y1, x2, y2], outline=COLOR_PICK_BOX, width=2)
        draw.text((x1, max(y1 - 12, 0)), "PICK", fill=COLOR_PICK_BOX)

    if ann.get("click_buffer") and len(ann["click_buffer"]) == 1:
        x, y = ann["click_buffer"][0]
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=COLOR_PICK_BOX, outline=COLOR_PICK_BOX)

    if ann.get("pick_point"):
        x, y = ann["pick_point"]
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=COLOR_AFFORDANCE, outline=COLOR_AFFORDANCE)
        draw.text((x + 8, y - 6), "AFF", fill=COLOR_AFFORDANCE)

    path = ann.get("gripper_path", [])
    if len(path) >= 2:
        for i in range(len(path) - 1):
            t = i / max(len(path) - 2, 1)
            r = int(COLOR_PATH_START[0] * (1 - t) + COLOR_PATH_END[0] * t)
            g = int(COLOR_PATH_START[1] * (1 - t) + COLOR_PATH_END[1] * t)
            b = int(COLOR_PATH_START[2] * (1 - t) + COLOR_PATH_END[2] * t)
            draw.line([path[i], path[i + 1]], fill=(r, g, b), width=2)
    for pt in path:
        draw.ellipse([pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3], fill=(255, 255, 255), outline=(0, 0, 0))

    return np.array(pil)


def format_annotation_tags(
    ann: Dict,
    scale: int = 1000,
    img_w: int = 256,
    img_h: int = 256,
    num_path_points: int = 5,
) -> str:
    """Generate copyable tag text from current annotations."""
    parts = []
    if ann.get("pick_box"):
        (x1, y1), (x2, y2) = ann["pick_box"]
        mx1, my1 = pixel_to_model(x1, y1, img_w, img_h, scale)
        mx2, my2 = pixel_to_model(x2, y2, img_w, img_h, scale)
        parts.append(f"<|box_start|>({mx1},{my1}),({mx2},{my2})<|box_end|>")

    if ann.get("pick_point"):
        px, py = ann["pick_point"]
        mx, my = pixel_to_model(px, py, img_w, img_h, scale)
        parts.append(f"<|affordance_2d_start|>({mx},{my})<|affordance_2d_end|>")

    path = ann.get("gripper_path", [])
    if len(path) >= 2:
        sampled = resample_path(path, num_path_points)
        pts = []
        for px, py in sampled:
            mx, my = pixel_to_model(px, py, img_w, img_h, scale)
            pts.append(f"({mx},{my})")
        parts.append(f"<|gripper_path_2d_start|>{';'.join(pts)}<|gripper_path_2d_end|>")

    return "\n".join(parts) if parts else ""


def annotation_status_text(ann: Dict) -> str:
    """Build a human-readable markdown summary of current annotations."""
    lines = []
    if ann.get("pick_box"):
        (x1, y1), (x2, y2) = ann["pick_box"]
        lines.append(f"- **Pick Box**: ({x1},{y1}) - ({x2},{y2})")
    if ann.get("pick_point"):
        x, y = ann["pick_point"]
        lines.append(f"- **Affordance Point**: ({x},{y})")
    path = ann.get("gripper_path", [])
    if path:
        lines.append(f"- **Gripper Path**: {len(path)} points")
    buf = ann.get("click_buffer", [])
    if buf:
        lines.append(f"- *(waiting for next click: {len(buf)} point buffered)*")
    return "\n".join(lines) if lines else "*No annotations*"


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load XVLA model and processor from checkpoint."""
    print(f"Loading config from: {config_path}")
    config = XVLAConfig.from_json_file(config_path)
    
    print(f"Creating model...")
    model = XVLA(config)
    
    print(f"Loading weights from: {checkpoint_path}")
    _load_checkpoint_keep_mismatch(model, checkpoint_path)
    
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    
    print(f"Building processor...")
    processor = build_xvla_processor(
        vlm_backbone_type="qwen3_vl",
        pretrained_name_or_path=getattr(config, "qwen3_pretrained", "Qwen/Qwen3-VL-2B-Instruct"),
        num_views=getattr(config, "num_views", 2),
        use_cot_training=True,
        cot_max_length=getattr(config, "cot_max_length", 768),
    )
    
    print("Model loaded successfully!")
    return model, processor, config


def _resize_image(img: Image.Image, size: tuple = (256, 256)) -> Image.Image:
    """Resize image to fixed resolution to match training data distribution."""
    return img.convert("RGB").resize(size, Image.LANCZOS)


def generate_cot(model, processor, image: Image.Image, instruction: str,
                 image2: Image.Image = None, device: str = "cuda",
                 diffusion_steps: int = 10):
    """Run CoT inference and return cot_text."""
    image = _resize_image(image)
    images = [image]
    if image2 is not None:
        images.append(_resize_image(image2))

    inputs = processor(images=images, language_instruction=instruction)

    def to_device(t):
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t)
        return t.to(device=device, dtype=torch.bfloat16) if t.is_floating_point() else t.to(device=device)

    inputs = {k: to_device(v) for k, v in inputs.items()}

    dim_proprio = getattr(model.action_space, "dim_proprio", model.action_space.dim_action)
    proprio = torch.zeros(1, dim_proprio, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        result = model.generate_actions(
            input_ids=inputs["input_ids"],
            image_input=inputs["image_input"],
            image_mask=inputs["image_mask"],
            image_grid_thw=inputs.get("image_grid_thw"),
            proprio=proprio,
            domain_id=None,
            steps=diffusion_steps,
            return_cot=True,
        )

    if isinstance(result, tuple):
        _, cot_texts = result
        cot_text = cot_texts[0]
    else:
        cot_text = ""

    return cot_text


def format_parsed_cot(elements: dict) -> str:
    """Format parsed CoT elements into readable markdown."""
    lines = []

    if elements.get("task"):
        lines.append(f"**Task:** {elements['task']}")
    if elements.get("subtasks"):
        lines.append(f"**Subtasks:** {' -> '.join(elements['subtasks'])}")
    if elements.get("current"):
        lines.append(f"**Current:** {elements['current']}")

    if lines:
        lines.append("")

    for box in elements.get("boxes", []):
        tag = box["tag"]
        label = box["label"]
        x1, y1, x2, y2 = box["coords"]
        lines.append(f"- **{tag}** `{label}` bbox=({x1},{y1}),({x2},{y2})")

    for pt in elements.get("points", []):
        x, y = pt["coords"]
        lines.append(f"- **{pt['tag']}** point=({x},{y})")

    for path in elements.get("paths", []):
        pts = path["points"]
        pts_str = " -> ".join(f"({x},{y})" for x, y in pts[:8])
        if len(pts) > 8:
            pts_str += f" ... ({len(pts)} points total)"
        lines.append(f"- **GRIPPER_PATH** {pts_str}")

    if not elements.get("boxes") and not elements.get("points") and not elements.get("paths"):
        lines.append("*No visual elements detected*")

    return "\n".join(lines)


def build_demo(model, processor, config, device: str = "cuda"):
    """Build the Gradio demo interface with interactive annotation support."""
    coord_scale = getattr(config, "cot_coord_scale", 1000)
    num_path_pts = getattr(config, "cot_gripper_future_steps", 5)

    # -- helpers that need closure over config ---------------------------------

    def _get_preview(image_np, ann):
        """Return image with user annotations drawn, resized to IMG_SIZE."""
        if image_np is None:
            return None
        pil = Image.fromarray(image_np) if isinstance(image_np, np.ndarray) else image_np
        frame = np.array(_resize_image(pil))
        return draw_annotations(frame, ann)

    def _make_tags(ann_state):
        return format_annotation_tags(
            ann_state, scale=coord_scale, img_w=IMG_SIZE[0], img_h=IMG_SIZE[1],
            num_path_points=num_path_pts,
        )

    # -- click handler ---------------------------------------------------------

    def on_image_click(image, mode, ann_state, evt: gr.SelectData):
        """Handle clicks on the input image to collect annotations."""
        if image is None or mode == "None":
            return image, ann_state, annotation_status_text(ann_state), _make_tags(ann_state)

        x, y = int(evt.index[0]), int(evt.index[1])
        h_disp, w_disp = image.shape[:2]
        x = int(x * IMG_SIZE[0] / w_disp)
        y = int(y * IMG_SIZE[1] / h_disp)
        x = max(0, min(x, IMG_SIZE[0] - 1))
        y = max(0, min(y, IMG_SIZE[1] - 1))

        if mode == "Pick Box":
            buf = ann_state.get("click_buffer", [])
            buf.append((x, y))
            if len(buf) >= 2:
                x1 = min(buf[0][0], buf[1][0])
                y1 = min(buf[0][1], buf[1][1])
                x2 = max(buf[0][0], buf[1][0])
                y2 = max(buf[0][1], buf[1][1])
                ann_state["pick_box"] = [(x1, y1), (x2, y2)]
                ann_state["click_buffer"] = []
            else:
                ann_state["click_buffer"] = buf

        elif mode == "Pick Point":
            ann_state["pick_point"] = (x, y)

        elif mode == "Gripper Path":
            ann_state.setdefault("gripper_path", []).append((x, y))

        preview = _get_preview(image, ann_state)
        return preview, ann_state, annotation_status_text(ann_state), _make_tags(ann_state)

    def clear_annotations(image, ann_state):
        ann_state = _empty_annotations()
        preview = _get_preview(image, ann_state)
        return preview, ann_state, annotation_status_text(ann_state), ""

    def on_mode_change(mode, ann_state):
        """Reset click buffer when switching modes."""
        ann_state["click_buffer"] = []
        return ann_state

    def on_image_upload(image, ann_state):
        """Reset annotations and show resized preview when a new image is uploaded."""
        ann_state = _empty_annotations()
        preview = _get_preview(image, ann_state)
        return preview, ann_state, annotation_status_text(ann_state), ""

    # -- predict ---------------------------------------------------------------

    def predict(image, instruction, image2, diffusion_steps, ann_state):
        if image is None:
            return None, "Please upload an image.", "", ann_state
        if not instruction or not instruction.strip():
            return None, "Please enter an instruction.", "", ann_state

        pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        pil_image2 = None
        if image2 is not None:
            pil_image2 = Image.fromarray(image2) if isinstance(image2, np.ndarray) else image2

        final_instruction = instruction.strip()

        t0 = time.time()
        cot_text = generate_cot(
            model, processor, pil_image, final_instruction,
            image2=pil_image2, device=device,
            diffusion_steps=int(diffusion_steps),
        )
        elapsed = time.time() - t0

        elements = parse_cot_elements(cot_text)

        frame = np.array(_resize_image(pil_image))
        frame = draw_annotations(frame, ann_state)
        annotated = visualize_cot_on_image(frame, cot_text, coord_scale=coord_scale)
        annotated_img = Image.fromarray(annotated)

        parsed_md = format_parsed_cot(elements)
        parsed_md = f"*Generation time: {elapsed:.2f}s*\n\n**Instruction sent:** `{final_instruction}`\n\n{parsed_md}"

        raw_display = cot_text.replace("<|", "\n<|") if cot_text else "(empty)"

        return annotated_img, parsed_md, raw_display, ann_state

    # -- layout ----------------------------------------------------------------

    with gr.Blocks(
        title="CoT Interactive Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        ann_state = gr.State(_empty_annotations())

        gr.Markdown("# CoT Interactive Demo")

        with gr.Accordion("Usage & Examples", open=False):
            gr.Markdown(
                "### How to use\n"
                "1. Upload an image, select an **Annotation Mode**, click on the image to annotate\n"
                "2. Copy the generated tags from **Generated Tags** box\n"
                "3. Paste the tags into the **Instruction** box to compose your full instruction\n"
                "4. Click **Generate CoT**\n\n"
                "### Instruction examples\n"
                "**Box** — replace the object name with the box tag:\n"
                "```\n"
                "pick <|box_start|>(316,512),(566,836)<|box_end|> into left\n"
                "```\n"
                "**Affordance** — insert after the object name:\n"
                "```\n"
                "pick the red cube with affordance <|affordance_2d_start|>(400,600)<|affordance_2d_end|> into left\n"
                "```\n"
                "**Gripper path** — append at the end:\n"
                "```\n"
                "pick the red cube, follow the gripper path <|gripper_path_2d_start|>(100,200);(200,300);(300,400);(400,500);(500,600)<|gripper_path_2d_end|>\n"
                "```\n"
                "**Combined** — you can mix multiple tags:\n"
                "```\n"
                "pick <|box_start|>(316,512),(566,836)<|box_end|> with affordance <|affordance_2d_start|>(400,600)<|affordance_2d_end|>, follow the gripper path <|gripper_path_2d_start|>(100,200);(200,300);(300,400);(400,500);(500,600)<|gripper_path_2d_end|>\n"
                "```"
            )

        with gr.Row():
            # ---- Left column: Input + Annotations ----------------------------
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                input_image = gr.Image(label="Main View (click to annotate)", type="numpy")
                annotation_preview = gr.Image(label="Annotation Preview (256x256)", type="numpy", interactive=False)
                input_image2 = gr.Image(label="Second View (optional)", type="numpy")

                instruction = gr.Textbox(
                    label="Instruction (paste tags here to compose full instruction)",
                    placeholder="e.g., pick <|box_start|>(316,512),(566,836)<|box_end|> into left",
                    lines=3,
                )
                diffusion_steps = gr.Slider(
                    minimum=1, maximum=20, value=10, step=1,
                    label="Diffusion Steps",
                )
                with gr.Accordion("Interactive Annotations", open=True):
                    ann_mode = gr.Radio(
                        choices=["None", "Pick Box", "Pick Point", "Gripper Path"],
                        value="None",
                        label="Annotation Mode",
                        info="Select a mode then click on the Main View image",
                    )
                    ann_status = gr.Markdown("*No annotations*")
                    ann_tags = gr.Textbox(
                        label="Generated Tags (copy & paste into Instruction)",
                        lines=3,
                        interactive=False,
                        show_copy_button=True,
                        placeholder="Tags will appear here after annotation — copy and paste into Instruction box",
                    )
                    clear_btn = gr.Button("Clear Annotations", variant="secondary")

                submit_btn = gr.Button("Generate CoT", variant="primary", size="lg")

            # ---- Right column: Output ----------------------------------------
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                output_image = gr.Image(label="CoT Visualization", type="pil")
                parsed_output = gr.Markdown(label="Parsed CoT Elements")
                with gr.Accordion("Raw CoT Text", open=False):
                    raw_output = gr.Textbox(label="Raw Model Output", lines=10, interactive=False)

        # -- event wiring ------------------------------------------------------

        input_image.select(
            fn=on_image_click,
            inputs=[input_image, ann_mode, ann_state],
            outputs=[annotation_preview, ann_state, ann_status, ann_tags],
        )

        input_image.change(
            fn=on_image_upload,
            inputs=[input_image, ann_state],
            outputs=[annotation_preview, ann_state, ann_status, ann_tags],
        )

        ann_mode.change(
            fn=on_mode_change,
            inputs=[ann_mode, ann_state],
            outputs=[ann_state],
        )

        clear_btn.click(
            fn=clear_annotations,
            inputs=[input_image, ann_state],
            outputs=[annotation_preview, ann_state, ann_status, ann_tags],
        )

        submit_btn.click(
            fn=predict,
            inputs=[input_image, instruction, input_image2, diffusion_steps, ann_state],
            outputs=[output_image, parsed_output, raw_output, ann_state],
        )

        gr.Markdown("---")
        gr.Markdown(
            "**Legend:** "
            "Green box = PICK, "
            "Red box = PLACE, "
            "Cyan box = OBJECTS, "
            "Yellow dot = AFFORDANCE, "
            "Green->Red path = GRIPPER trajectory | "
            "White dots = user gripper path waypoints"
        )

    return demo


def main():
    parser = argparse.ArgumentParser("CoT Pretrain Verification Demo")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to XVLA checkpoint directory")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to XVLA config JSON")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda / cpu)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    args = parser.parse_args()

    model, processor, config = load_model(args.checkpoint, args.config, args.device)
    demo = build_demo(model, processor, config, args.device)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
