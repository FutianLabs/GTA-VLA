#!/usr/bin/env python3
"""
Run SimplerEnv WidowX evaluation directly against a locally loaded VLA model
(no FastAPI server required). Supports GTA-VLA as well as OpenVLA, OpenVLA-OFT,
and VLA-Adapter checkpoints. Mirrors the behaviour of the HTTP clients under
evaluation/simpler/WidowX but keeps the model in memory.
"""
import argparse
import json
import logging
import math
import os
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from sapien.core import Pose
from transformers import AutoProcessor
from scipy.spatial.transform import Rotation as R

# Headless rendering required before importing simpler_env
# Note: Environment variables should be set before Python starts for best results
# Use scripts/run_widowx_eval.sh wrapper script
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("MUJOCO_GL", "egl")  # Use EGL for headless rendering
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # PyOpenGL EGL backend
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")  # Specify GPU device

# Additional environment variables for headless rendering
os.environ.setdefault("SAPIEN_RENDERER", "offscreen")  # Force offscreen rendering in Sapien
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")  # Override OpenGL version if needed

import simpler_env  # noqa: E402
from simpler_env.utils.env.observation_utils import (  # noqa: E402
    get_image_from_maniskill2_obs_dict,
)
from simpler_env.utils.visualization import write_video  # noqa: E402

from models.modeling_gtavla import GTAVLA  # noqa: E402
from models.processing_gtavla import GTAVLAProcessor, build_gtavla_processor  # noqa: E402

try:
    from peft import PeftModel
except ImportError:  # Optional dependency
    PeftModel = None

logger = logging.getLogger("evaluate_widowx")


def _normalize_arch(name: str) -> str:
    name = name.lower()
    if name in {"xvla", "gtavla", "gta-vla", "gta_vla"}:
        return "gtavla"
    if name in {"openvla_oft", "openvlaoft"}:
        return "openvla-oft"
    if name in {"vla_adapter", "vlaadapter"}:
        return "vla-adapter"
    return name


# -----------------------------------------------------------------------------#
# Geometry helpers                                                             #
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
def rotate6D_to_euler_xyz(v6: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation back to Euler angles (xyz).

    Matches the logic used by the existing widowx client scripts.
    """
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError(f"Last dimension must be 6, got {v6.shape[-1]}")
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)
    # scipy.spatial.transform is heavier to import here; direct conversion is sufficient
    from scipy.spatial.transform import Rotation as R

    return R.from_matrix(rot_mats).as_euler("xyz")


def build_initial_proprio(obs: Dict) -> np.ndarray:
    """
    Compose the proprioception vector used by the original widowx clients.
    """
    ee_pose_wrt_base = Pose(
        p=obs["agent"]["base_pose"][:3],
        q=obs["agent"]["base_pose"][3:],
    ).inv() * Pose(
        p=obs["extra"]["tcp_pose"][:3],
        q=obs["extra"]["tcp_pose"][3:],
    )
    # Compute actual rot6d from environment quaternion
    q_wxyz = ee_pose_wrt_base.q
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)
    rot_mat = R.from_quat(q_xyzw).as_matrix()
    rot6d = rot_mat[:, :2].flatten().astype(np.float32)  # first two columns, row-major
    gripper_open = np.array([0.0], dtype=np.float32)  # default open
    proprio = torch.from_numpy(
        np.concatenate([ee_pose_wrt_base.p, rot6d, gripper_open])
    ).to(dtype=torch.float32)
    proprio = torch.cat([proprio, torch.zeros_like(proprio)], dim=-1).numpy()
    return proprio


def get_model_images_from_obs(obs: Dict, num_views: int) -> List[np.ndarray]:
    images = [obs["image"]["3rd_view_camera"]["rgb"]]
    if num_views > 1 and obs.get("image") and "wrist_camera" in obs["image"]:
        images.append(obs["image"]["wrist_camera"]["rgb"])
    return images


# -----------------------------------------------------------------------------#
# Local VLA policy wrapper                                                     #
# -----------------------------------------------------------------------------#
class LocalWidowXAgent:
    """
    Thin policy wrapper that calls an in-memory VLA model (GTA-VLA/OpenVLA variants)
    to produce actions. Behaviour mirrors the HTTP-based clients in
    evaluation/simpler/WidowX.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        device: torch.device,
        denoising_steps: int = 10,
        domain_id: int = 0,
        model_arch: str = "gtavla",
        unnorm_key: Optional[str] = None,
        save_video: bool = False,
        z_offset_m: float = 0.0,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.denoising_steps = denoising_steps
        self.domain_id = torch.tensor([domain_id], device=self.device, dtype=torch.long)
        self.model_arch = _normalize_arch(model_arch)
        self.unnorm_key = unnorm_key
        self.z_offset_m = z_offset_m
        self.dtype = next(model.parameters()).dtype
        
        # CoT support: auto-enable if model has CoT training and we're saving videos
        self.use_cot = getattr(model.config, 'use_cot_training', False) if hasattr(model, 'config') else False
        self.enable_cot_visualization = self.use_cot and save_video
        if self.enable_cot_visualization:
            logger.info("CoT visualization enabled (model has use_cot_training=True and save_video=True)")
        
        # Track frame type for visualization
        self.is_model_input_frame = False
        
        # Statistics tracking
        self.total_model_queries = 0
        self.total_frames = 0
        
        self.reset(np.zeros(14, dtype=np.float32), "")

    def reset(self, proprio: np.ndarray, instruction: str) -> None:
        self.proprio = np.asarray(proprio, dtype=np.float32)
        self.instruction = instruction
        self.action_plan: Deque[List[float]] = deque()
        self.last_cot_text: Optional[str] = None  # Store last CoT output for visualization
        self.is_model_input_frame = False
        self.last_raw_action: Optional[np.ndarray] = None
        self.last_env_action: Optional[np.ndarray] = None

    def _prepare_gtavla_inputs(self, images: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        pil_imgs = [Image.fromarray(image).resize((256, 256), Image.LANCZOS) for image in images]
        processed = self.processor(
            images=pil_imgs,
            language_instruction=self.instruction,
        )

        def to_device(t: torch.Tensor) -> torch.Tensor:
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            if t.is_floating_point():
                return t.to(device=self.device, dtype=self.dtype)
            return t.to(device=self.device)

        inputs = {k: to_device(v) for k, v in processed.items()}
        inputs.update(
            {
                "proprio": torch.as_tensor(self.proprio, device=self.device, dtype=self.dtype).unsqueeze(0),
                "domain_id": self.domain_id,
            }
        )
        return inputs

    def _enqueue_actions(self, raw_action: np.ndarray) -> None:
        actions = np.asarray(raw_action)
        if actions.ndim == 1:
            actions = actions[None, :]
        self.action_plan.extend(actions.tolist())

    def _queue_plan(self, images: List[np.ndarray]) -> None:
        if self.model_arch == "gtavla":
            inputs = self._prepare_gtavla_inputs(images)
            with torch.inference_mode():
                # Generate actions with optional CoT
                if self.enable_cot_visualization and self.use_cot:
                    result = self.model.generate_actions(
                        steps=self.denoising_steps, return_cot=True, **inputs
                    )
                    action_tensor, cot_texts = result
                    action = action_tensor.squeeze(0).float().cpu().numpy()
                    self.last_cot_text = cot_texts[0] if cot_texts else None
                else:
                    action = (
                        self.model.generate_actions(steps=self.denoising_steps, **inputs)
                        .squeeze(0)
                        .float()
                        .cpu()
                        .numpy()
                    )
        else:
            with torch.inference_mode():
                action = self.model.generate_actions(
                    instruction=self.instruction,
                    images=[Image.fromarray(images[0])],
                    processor=self.processor,
                    unnorm_key=self.unnorm_key,
                )
        self._enqueue_actions(action)

    def _convert_action_for_env(self, action_pred: np.ndarray, gripper_close_threshold: float) -> np.ndarray:
        action_pred = np.asarray(action_pred, dtype=np.float32)
        action_xyz = action_pred[:3].copy()
        action_xyz[2] += self.z_offset_m

        if action_pred.shape[0] >= 9:
            euler = rotate6D_to_euler_xyz(action_pred[3:9])
        elif action_pred.shape[0] >= 6:
            from scipy.spatial.transform import Rotation as R

            euler = R.from_rotvec(action_pred[3:6]).as_euler("xyz")
        else:
            raise ValueError(f"Unsupported action shape {action_pred.shape}")

        grip_source = action_pred[9] if action_pred.shape[0] > 9 else action_pred[-1]
        gripper = 1.0 if grip_source < gripper_close_threshold else -1.0

        return np.concatenate([action_xyz, euler, np.array([gripper], dtype=np.float32)])

    def step(self, images: List[np.ndarray], gripper_close_threshold: float) -> np.ndarray:
        self.total_frames += 1
        
        # Check if we need to query the model
        if not self.action_plan:
            self.is_model_input_frame = True
            self.total_model_queries += 1
            self._queue_plan(images)
        else:
            self.is_model_input_frame = False

        action_pred = np.array(self.action_plan.popleft(), dtype=np.float32)
        copy_len = min(action_pred.shape[0], self.proprio.shape[0])
        self.proprio[:copy_len] = action_pred[:copy_len]
        env_action = self._convert_action_for_env(action_pred, gripper_close_threshold)
        self.last_raw_action = action_pred.copy()
        self.last_env_action = env_action.copy()
        return env_action
    
    def get_last_cot_text(self) -> Optional[str]:
        """Get the last generated CoT text (only available when enable_cot_visualization=True)."""
        return self.last_cot_text
    
    def get_is_model_input_frame(self) -> bool:
        """Get whether the current frame was input to the model (vs cached action)."""
        return self.is_model_input_frame

    def get_last_actions(self) -> Dict[str, Optional[np.ndarray]]:
        return {
            "raw_action": self.last_raw_action,
            "env_action": self.last_env_action,
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Get agent statistics (model queries, cache hit rate, etc.)."""
        if self.total_frames == 0:
            return {"model_queries": 0, "total_frames": 0, "cache_rate": 0.0, "avg_actions_per_query": 0.0}
        
        cache_rate = (self.total_frames - self.total_model_queries) / self.total_frames
        avg_actions = self.total_frames / max(1, self.total_model_queries)
        
        return {
            "model_queries": self.total_model_queries,
            "total_frames": self.total_frames,
            "cache_rate": cache_rate,
            "avg_actions_per_query": avg_actions,
        }


# -----------------------------------------------------------------------------#
# Model loading                                                                #
# -----------------------------------------------------------------------------#
def load_model_and_processor(
    model_path: str,
    processor_path: Optional[str],
    lora_path: Optional[str],
    device: torch.device,
    model_arch: str,
) -> tuple[torch.nn.Module, Any]:
    def resolve_processor_path(path: Optional[str]) -> str:
        if path and Path(path).exists():
            return path
        if path and path != model_path and not Path(path).exists():
            logger.warning("Processor path %s not found, falling back to model path %s", path, model_path)
        return model_path

    arch = _normalize_arch(model_arch)
    processor_resolved = resolve_processor_path(processor_path)

    if arch == "gtavla":
        # Load model first to determine backbone type
        model = GTAVLA.from_pretrained(model_path)
        
        # Determine processor based on backbone type (same logic as libero evaluation)
        vlm_type = getattr(model.config, "vlm_backbone_type", "florence2")
        
        if vlm_type == "qwen3_vl":
            qwen3_path = getattr(model.config, "qwen3_pretrained", "Qwen/Qwen3-VL-2B-Instruct")
            use_cot_training = getattr(model.config, "use_cot_training", False)
            num_views = int(getattr(model.config, "num_views", 1))
            processor = build_gtavla_processor(
                vlm_backbone_type="qwen3_vl",
                pretrained_name_or_path=qwen3_path,
                num_views=num_views,
                use_cot_training=use_cot_training,
            )
            logger.info(f"Loaded Qwen3-VL processor (num_views={num_views}, cot={use_cot_training})")
        else:
            # Load Florence2 processor from checkpoint
            processor = GTAVLAProcessor.from_pretrained(processor_resolved)
            processor.num_views = int(getattr(model.config, "num_views", getattr(processor, "num_views", 1)))
            logger.info(f"Loaded Florence2 processor from {processor_resolved}")
        
        uses_flash_attn = (
            vlm_type == 'qwen3_vl' and getattr(model.config, 'qwen3_use_flash_attn', False)
        )
        
        if uses_flash_attn:
            # FlashAttention requires bf16/fp16
            model = model.to(device).to(torch.bfloat16)
            logger.info(f"Using bfloat16 for {vlm_type} with FlashAttention")
        else:
            model = model.to(device).to(torch.float32)

        if lora_path:
            if PeftModel is None:
                raise ImportError("peft is required for loading LoRA weights.")
            model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float32).to(device)

        model.eval()
        return model, processor

    from models.openvla.modeling_openvla import OpenVLA, OpenVLAAdapter, OpenVLAOFT
    model_cls_map = {
        "openvla": OpenVLA,
        "openvla-oft": OpenVLAOFT,
        "vla-adapter": OpenVLAAdapter,
    }
    if arch not in model_cls_map:
        raise ValueError(f"Unsupported model_arch {arch}.")

    model_cls = model_cls_map[arch]
    model = model_cls.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
    try:
        processor = AutoProcessor.from_pretrained(processor_resolved, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - best effort fallback
        logger.warning("Falling back to model.processor because AutoProcessor load failed: %s", exc)
        processor = getattr(model, "processor", None)
    if processor is None:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if getattr(model, "processor", None) is None:
        try:
            model.processor = processor
        except Exception:
            pass
    if lora_path:
        logger.warning("Ignoring lora_path for %s models; LoRA loading only supported for GTA-VLA.", arch)
    model.eval()
    return model, processor


# -----------------------------------------------------------------------------#
# Evaluation loop                                                              #
# -----------------------------------------------------------------------------#



def evaluate_task(
    agent: LocalWidowXAgent,
    task: str,
    output_dir: Path,
    episodes: int,
    seed_offset: int,
    save_video: bool,
    TASK_CONFIGS: Dict[str, Dict[str, float]],
    enable_cot_visualization: bool = False,
    action_dump_dir: Optional[Path] = None,
) -> float:
    # Lazy import CoT visualization utilities
    cot_visualizer = None
    if enable_cot_visualization:
        try:
            from scripts.visualize_cot import visualize_cot_on_image
            cot_visualizer = visualize_cot_on_image
        except ImportError:
            logger.warning("CoT visualization requested but scripts.visualize_cot not found")
            enable_cot_visualization = False
    cfg = TASK_CONFIGS[task]
    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    action_task_dir = None
    if action_dump_dir is not None:
        action_task_dir = action_dump_dir / task
        action_task_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    for episode in range(episodes):
        env = simpler_env.make(task, renderer_kwargs={"offscreen_only": True})
        obs, _ = env.reset(options={"obj_init_options": {"episode_id": episode + seed_offset}})
        instruction = env.get_language_instruction()
        proprio = build_initial_proprio(obs)
        agent.reset(proprio, instruction)

        frames: List[np.ndarray] = []
        done = False
        reward = 0.0
        step_idx = 0
        last_cot_text = None
        last_is_input = False
        action_log_file = None
        if action_task_dir is not None:
            action_log_path = action_task_dir / f"episode_{episode + seed_offset:03d}.jsonl"
            action_log_file = action_log_path.open("w", encoding="utf-8")
        for step_idx in range(cfg["max_steps"]):
            image = get_image_from_maniskill2_obs_dict(env, obs)
            model_images = get_model_images_from_obs(obs, int(getattr(agent.processor, "num_views", 1)))
            action = agent.step(model_images, gripper_close_threshold=cfg["gripper_close_threshold"])
            obs, reward, done, _, _ = env.step(action)
            if action_log_file is not None:
                action_record = agent.get_last_actions()
                action_log_file.write(json.dumps({
                    "episode": episode + seed_offset,
                    "step": step_idx,
                    "model_input_frame": agent.get_is_model_input_frame(),
                    "raw_action": action_record["raw_action"].tolist() if action_record["raw_action"] is not None else None,
                    "env_action": action_record["env_action"].tolist() if action_record["env_action"] is not None else None,
                    "reward": float(reward),
                    "done": bool(done),
                }) + "\n")
            if save_video:
                frame = image.copy()
                # Apply CoT visualization if enabled
                if enable_cot_visualization and cot_visualizer is not None:
                    cot_text = agent.get_last_cot_text()
                    is_input = agent.get_is_model_input_frame()
                    
                    # Debug: Print CoT for every model input step
                    if is_input and cot_text:
                        logger.info(f"[ep{episode} step{step_idx}] CoT:\n{cot_text}")
                    
                    frame = cot_visualizer(frame, cot_text, is_model_input_frame=is_input)
                    # Store for final frame
                    last_cot_text = cot_text
                    last_is_input = is_input
                frames.append(frame)
            if done:
                # Capture the final success state frame (image after the last action)
                if save_video:
                    final_image = get_image_from_maniskill2_obs_dict(env, obs)
                    final_frame = final_image.copy()
                    if enable_cot_visualization and cot_visualizer is not None:
                        final_frame = cot_visualizer(final_frame, last_cot_text, is_model_input_frame=False)
                    # Add a few frames of the final success state for better visualization
                break

        steps_taken = step_idx + 1 if done else cfg["max_steps"]
        successes += int(done)        
        if save_video:
            video_path = task_dir / f"{task}_ep{episode}_success{int(done)}.mp4"
            print(f"Saving video to {video_path}")
            write_video(str(video_path), frames, fps=10)
        if action_log_file is not None:
            action_log_file.close()
        env.close()

        logger.info(
            "[%s] episode=%d success=%s steps=%d/%d success_rate=%.3f reward=%.3f (%d/%d)",
            task,
            episode,
            bool(done),
            steps_taken,
            cfg["max_steps"],
            successes / float(episode + 1),
            reward,
            successes,
            episode + 1,
        )

    return successes / float(episodes)


def run_widowx_eval(
    model: torch.nn.Module,
    processor: Any,
    save_path: Path,
    tasks: Iterable[str],
    episodes: int,
    seed_offset: int,
    denoising_steps: int,
    domain_id: int,
    device: Optional[torch.device] = None,
    save_video: bool = True,
    model_arch: str = "gtavla",
    openvla_unnorm_key: Optional[str] = None,
    TASK_CONFIGS = None,
    action_dump_dir: Optional[Path] = None,
    z_offset_m: float = 0.0,
) -> Dict[str, float]:
    device = device or next(model.parameters()).device
    agent = LocalWidowXAgent(
        model=model,
        processor=processor,
        device=device,
        denoising_steps=denoising_steps,
        domain_id=domain_id,
        model_arch=model_arch,
        unnorm_key=openvla_unnorm_key,
        save_video=save_video,
        z_offset_m=z_offset_m,
    )
    save_path.mkdir(parents=True, exist_ok=True)

    results: Dict[str, float] = {}
    for task in tasks:
        results[task] = evaluate_task(
            agent=agent,
            task=task,
            output_dir=save_path,
            episodes=episodes,
            seed_offset=seed_offset,
            save_video=save_video,
            TASK_CONFIGS=TASK_CONFIGS,
            enable_cot_visualization=agent.enable_cot_visualization,
            action_dump_dir=action_dump_dir,
        )
    return results


# -----------------------------------------------------------------------------#
# CLI                                                                          #
# -----------------------------------------------------------------------------#
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Direct WidowX evaluation (no server needed)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint directory.")
    parser.add_argument(
        "--model_arch",
        type=str,
        default="gtavla",
        choices=["gtavla", "gta-vla", "gta_vla", "xvla", "openvla", "openvla-oft", "openvla_oft", "openvlaoft", "vla-adapter", "vla_adapter", "vlaadapter"],
        help="Which model family to evaluate.",
    )
    parser.add_argument("--processor_path", type=str, default=None, help="Optional processor path (defaults to model).")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA weights to merge for evaluation.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"],
        choices=["widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"],
        help="WidowX tasks to evaluate.",
    )
    parser.add_argument("--output_dir", type=str, default="evaluation_outputs/widowx", help="Directory for eval logs/videos.")
    parser.add_argument("--episodes", type=int, default=24, help="Episodes per task.")
    parser.add_argument("--seed_offset", type=int, default=0, help="Offset for episode ids passed to env reset.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for model inference.")
    parser.add_argument("--denoising_steps", type=int, default=10, help="Diffusion denoising steps.")
    parser.add_argument("--domain_id", type=int, default=0, help="Domain id used by the model.")
    parser.add_argument("--eval_setting", type=str, default="cogact", choices=["gtavla", "xvla", "official", "cogact"], help="Evaluation setting.")
    parser.add_argument(
        "--openvla_unnorm_key",
        type=str,
        default=None,
        help="De-normalization key for OpenVLA action stats when multiple datasets are present.",
    )
    parser.add_argument(
        "--save_video",
        dest="save_video",
        action="store_true",
        default=True,
        help="Save evaluation rollouts as videos (default: on).",
    )
    parser.add_argument(
        "--no_save_video",
        dest="save_video",
        action="store_false",
        help="Disable saving videos to speed up evaluation.",
    )
    parser.add_argument("--action_dump_dir", type=str, default=None, help="Directory to dump per-step raw/env actions.")
    parser.add_argument("--z_offset_cm", type=float, default=0.0, help="Additive z offset in centimeters.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max environment steps per episode for all tasks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    arch = _normalize_arch(args.model_arch)
    logger.info("Loading %s model on device %s", arch, device)
    model, processor = load_model_and_processor(args.model_path, args.processor_path, args.lora_path, device, arch)

    out_dir = Path(args.output_dir)
    logger.info(
        "Starting WidowX evaluation | arch=%s | tasks=%s | episodes=%s | save_dir=%s",
        arch,
        args.tasks,
        args.episodes,
        out_dir,
    )
    if args.eval_setting in {"xvla", "gtavla"}:
        TASK_CONFIGS = {
            "widowx_spoon_on_towel": {"max_steps": 1200, "gripper_close_threshold": 0.7},
            "widowx_carrot_on_plate": {"max_steps": 1200, "gripper_close_threshold": 0.95},
            "widowx_stack_cube": {"max_steps": 1200, "gripper_close_threshold": 0.91},
            "widowx_put_eggplant_in_basket": {"max_steps": 1200, "gripper_close_threshold": 0.8},
        }
    elif args.eval_setting == "official":
        TASK_CONFIGS = {
            "widowx_spoon_on_towel": {"max_steps": 60, "gripper_close_threshold": 0.7},
            "widowx_carrot_on_plate": {"max_steps": 60, "gripper_close_threshold": 0.95},
            "widowx_stack_cube": {"max_steps": 60, "gripper_close_threshold": 0.91},
            "widowx_put_eggplant_in_basket": {"max_steps": 120, "gripper_close_threshold": 0.8},
        }
    elif args.eval_setting == "cogact":
        TASK_CONFIGS = {
            "widowx_spoon_on_towel": {"max_steps": 120, "gripper_close_threshold": 0.7},
            "widowx_carrot_on_plate": {"max_steps": 120, "gripper_close_threshold": 0.95},
            "widowx_stack_cube": {"max_steps": 120, "gripper_close_threshold": 0.91},
            "widowx_put_eggplant_in_basket": {"max_steps": 120, "gripper_close_threshold": 0.8},
        }
    if args.max_steps is not None:
        for cfg in TASK_CONFIGS.values():
            cfg["max_steps"] = args.max_steps
    results = run_widowx_eval(
        model=model,
        processor=processor,
        save_path=out_dir,
        tasks=args.tasks,
        episodes=args.episodes,
        seed_offset=args.seed_offset,
        denoising_steps=args.denoising_steps,
        domain_id=args.domain_id,
        device=device,
        save_video=args.save_video,
        model_arch=arch,
        openvla_unnorm_key=args.openvla_unnorm_key,
        TASK_CONFIGS=TASK_CONFIGS,
        action_dump_dir=Path(args.action_dump_dir) if args.action_dump_dir else None,
        z_offset_m=args.z_offset_cm / 100.0,
    )
    
    # Print detailed results
    print("\n" + "=" * 80)
    print("📊 WidowX Evaluation Results Summary")
    print("=" * 80)
    for task_name, success_rate in results.items():
        print(f"  {task_name:40s}: {success_rate:6.2%} ({int(success_rate * args.episodes)}/{args.episodes})")
    print("-" * 80)
    
    # Calculate and print overall average
    overall_avg = sum(results.values()) / len(results) if results else 0.0
    total_success = sum(int(sr * args.episodes) for sr in results.values())
    total_episodes = len(results) * args.episodes
    print(f"  {'Overall Average':40s}: {overall_avg:6.2%} ({total_success}/{total_episodes})")
    print("=" * 80)
    
    logger.info("Evaluation finished: %s", results)


if __name__ == "__main__":
    main()
