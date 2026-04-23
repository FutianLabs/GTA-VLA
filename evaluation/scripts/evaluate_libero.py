#!/usr/bin/env python3
"""
Run LIBERO evaluation directly against a locally loaded VLA model (no HTTP server).

Supports GTA-VLA as well as OpenVLA, OpenVLA-OFT, and VLA-Adapter checkpoints. This
script can be used as a standalone evaluator or imported from training to trigger
periodic evaluations.
"""
import argparse
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

# Import will be done dynamically based on --sim argument
from models.modeling_gtavla import GTAVLA
from models.processing_gtavla import GTAVLAProcessor, build_gtavla_processor
try:
    from peft import PeftModel
except ImportError:  # Optional dependency
    PeftModel = None

logger = logging.getLogger("evaluate")


def _import_libero_client(sim_type: str):
    """
    Dynamically import the appropriate libero client module based on sim_type.
    
    Args:
        sim_type: Either 'libero' or 'libero_plus'
    
    Returns:
        Module containing LiberoAbsActionProcessor, _flip_agentview, and eval_libero
    """
    if sim_type == "libero":
        from evaluation.libero import libero_client as client_module
    elif sim_type == "libero_plus":
        from evaluation.libero import libero_plus_client as client_module
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}. Must be 'libero' or 'libero_plus'")
    
    return client_module


class EvalRunResult(dict):
    """
    Success rates per suite plus average steps (including max horizon for failures).
    Behaves like a dict keyed by suite name for backward compatibility.
    """

    def __init__(self, success_rates: Dict[str, float], avg_steps: Dict[str, float]):
        super().__init__(success_rates)
        self.avg_steps = avg_steps

    def __repr__(self) -> str:
        return f"EvalRunResult(success_rate={dict(self)}, avg_steps={self.avg_steps})"




def _normalize_arch(name: str) -> str:
    name = name.lower()
    if name in {"xvla", "gtavla", "gta-vla", "gta_vla"}:
        return "gtavla"
    if name in {"openvla_oft", "openvlaoft"}:
        return "openvla-oft"
    if name in {"vla_adapter", "vlaadapter"}:
        return "vla-adapter"
    return name


class LocalVLAAgent:
    """
    Thin policy wrapper that calls an in-memory VLA model (GTA-VLA or OpenVLA variants)
    to produce actions. Mirrors the HTTP client behaviour used by the FastAPI server.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        device: torch.device,
        denoising_steps: int = 10,
        domain_id: int = 3,
        model_arch: str = "gtavla",
        unnorm_key: Optional[str] = None,
        gripper_close_threshold: float = 0.5,
        client_module: Any = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.denoising_steps = denoising_steps
        self.domain_id = torch.tensor([domain_id], device=self.device, dtype=torch.long)
        self.model_arch = _normalize_arch(model_arch)
        self.unnorm_key = unnorm_key
        self.gripper_close_threshold = gripper_close_threshold
        
        vlm_type = getattr(model.config, 'vlm_backbone_type', 'florence2') if hasattr(model, 'config') else 'florence2'
        self.use_qwen_template = vlm_type == 'qwen3_vl'
        if self.use_qwen_template:
            logger.info(f"Using {vlm_type} chat template for instructions")
        
        # Import client module functions/classes
        if client_module is None:
            # Default to libero_client for backward compatibility
            from evaluation.libero import libero_client as client_module
        self.client_module = client_module
        self.rot_processor = client_module.LiberoAbsActionProcessor()
        self._flip_agentview = client_module._flip_agentview
        
        self.dtype = next(model.parameters()).dtype
        self.reset()

    def reset(self) -> None:
        self.proprio: Optional[np.ndarray] = None  # [20] absolute ee + copy
        self.action_plan: deque[List[float]] = deque()

    def _extract_images(self, obs: Dict) -> List[Image.Image]:
        main_view = self._flip_agentview(obs["agentview_image"])
        wrist_view = obs["robot0_eye_in_hand_image"]
        return [Image.fromarray(main_view), Image.fromarray(wrist_view)]

    def _prepare_gtavla_inputs(self, obs: Dict, goal: str) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for GTA-VLA model inference.
        
        For Qwen3-VL backbone, the processor will automatically apply the chat template:
        <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>instruction<|im_end|>
        
        For Florence2 backbone, the processor uses the raw instruction directly.
        """
        images = self._extract_images(obs)

        if self.proprio is None:
            closed_loop = np.concatenate(
                [obs["robo_pos"], obs["robo_ori"], np.array([0.0])], axis=-1
            )
            self.proprio = np.concatenate([closed_loop, np.zeros_like(closed_loop)], axis=-1)

        # The processor will automatically apply the correct template based on vlm_backbone_type
        # No need to manually format the instruction here
        processed = self.processor(images=images, language_instruction=goal)
        
        if self.use_qwen_template:
            # Log the formatted instruction for debugging (first time only)
            if not hasattr(self, '_logged_template'):
                logger.debug(f"Qwen formatted instruction (sample): {goal[:50]}...")
                self._logged_template = True

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

    def _convert_actions(self, actions: np.ndarray) -> np.ndarray:
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2:
            raise ValueError(f"Unsupported action shape {actions.shape}, expected [T, D] or [D].")

        dim = actions.shape[1]
        if dim >= 9:
            target_eef = actions[:, :3]
            target_axis = self.rot_processor.Rotate6D_to_AxisAngle(actions[:, 3:9])
            target_grip = actions[:, 9:10] if dim > 9 else actions[:, -1:]
        elif dim >= 7:
            target_eef = actions[:, :3]
            target_axis = actions[:, 3:6]
            target_grip = actions[:, 6:7]
        else:
            raise ValueError(f"Expected at least 7 action dims, got {dim}.")

        return np.concatenate([target_eef, target_axis, target_grip], axis=-1)

    def _queue_plan(self, raw_action: np.ndarray) -> None:
        actions = np.asarray(raw_action)
        if actions.ndim == 1:
            actions = actions[None, :]

        if self.model_arch == "gtavla":
            if self.proprio is None:
                self.proprio = np.zeros_like(actions[-1])
            copy_len = min(9, actions.shape[1], self.proprio.shape[0])
            self.proprio[:copy_len] = actions[-1, :copy_len].copy()

        final_action = self._convert_actions(actions)
        for row in final_action.tolist():
            self.action_plan.append(row)

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            if self.model_arch == "gtavla":
                inputs = self._prepare_gtavla_inputs(obs, goal)
                with torch.inference_mode():
                    action = (
                        self.model.generate_actions(steps=self.denoising_steps, **inputs)
                        .squeeze(0)
                        .float()
                        .cpu()
                        .numpy()
                    )
            else:
                images = self._extract_images(obs)
                with torch.inference_mode():
                    action = self.model.generate_actions(
                        instruction=goal,
                        images=images,
                        processor=self.processor,
                        unnorm_key=self.unnorm_key,
                    )
            self._queue_plan(np.asarray(action))

        action_predict = np.array(self.action_plan.popleft(), dtype=np.float32)
        action_predict[-1] = 1.0 if action_predict[-1] > self.gripper_close_threshold else -1.0
        return action_predict


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
        
        # Determine processor based on backbone type (same logic as vla_factory.py)
        vlm_type = getattr(model.config, "vlm_backbone_type", "florence2")
        
        if vlm_type == "qwen3_vl":
            qwen3_path = getattr(model.config, "qwen3_pretrained", "Qwen/Qwen3-VL-2B-Instruct")
            use_cot_training = getattr(model.config, "use_cot_training", False)
            processor = build_gtavla_processor(
                vlm_backbone_type="qwen3_vl",
                pretrained_name_or_path=qwen3_path,
                num_views=2,  # LIBERO uses 2 camera views
                use_cot_training=use_cot_training,
            )
            logger.info(f"Loaded Qwen3-VL processor (num_views=2, cot={use_cot_training})")
        else:
            # Load Florence2 processor from checkpoint
            processor = GTAVLAProcessor.from_pretrained(processor_resolved)
            logger.info(f"Loaded Florence2 processor from {processor_resolved}")
        
        uses_flash_attn = (
            vlm_type == 'qwen3_vl' and getattr(model.config, 'qwen3_use_flash_attn', False)
        )
        
        if uses_flash_attn:
            # FlashAttention requires bf16/fp16
            model = model.to(device).to(torch.bfloat16)
            logger.info(f"Using bfloat16 for {vlm_type} with FlashAttention")
        else:
            # Other configurations can use fp32
            model = model.to(device).to(torch.float32)
            logger.info("Using float32 for evaluation")
        
        if lora_path:
            if PeftModel is None:
                raise ImportError("peft is required for loading LoRA weights.")
            model = PeftModel.from_pretrained(model, lora_path).to(device)
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


def run_libero_eval(
    model: torch.nn.Module,
    processor: Any,
    save_path: Path,
    task_suites: Iterable[str],
    num_episodes: int = 10,
    init_seed: int = 42,
    act_type: str = "abs",
    denoising_steps: int = 10,
    domain_id: int = 3,
    device: Optional[torch.device] = None,
    save_video: bool = True,
    model_arch: str = "gtavla",
    openvla_unnorm_key: Optional[str] = None,
    gripper_close_threshold: float = 0.5,
    sim_type: str = "libero",
    max_tasks: Optional[int] = None,
) -> EvalRunResult:
    # Import the appropriate client module based on sim_type
    client_module = _import_libero_client(sim_type)
    
    device = device or next(model.parameters()).device
    agent = LocalVLAAgent(
        model=model,
        processor=processor,
        device=device,
        denoising_steps=denoising_steps,
        domain_id=domain_id,
        model_arch=model_arch,
        unnorm_key=openvla_unnorm_key,
        gripper_close_threshold=gripper_close_threshold,
        client_module=client_module,
    )
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Use eval_libero from the appropriate client module
    success_rates, avg_steps = client_module.eval_libero(
        agent=agent,
        save_path=save_path,
        num_episodes=num_episodes,
        init_seed=init_seed,
        act_type=act_type,
        task_suites=task_suites,
        save_video=save_video,
        max_tasks=max_tasks,
    )
    return EvalRunResult(success_rates, avg_steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Direct LIBERO evaluation (no server needed)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint directory.")
    parser.add_argument(
        "--sim",
        type=str,
        default="libero",
        choices=["libero", "libero_plus"],
        help="Simulation type: 'libero' for standard LIBERO evaluation, 'libero_plus' for LIBERO-Plus evaluation with fine-grained perturbation types.",
    )
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
        "--task_suites",
        nargs="+",
        default=None,
        help="LIBERO suites to evaluate.",
    )
    parser.add_argument("--output_dir", type=str, default="evaluation_outputs", help="Directory for eval logs/videos.")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per task.")
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to evaluate per suite. Use for quick testing (e.g., --max_tasks 20). Default: all tasks.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed.")
    parser.add_argument("--act_type", type=str, default="abs", choices=["abs", "rel"], help="Action type for env.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for model inference.")
    parser.add_argument("--denoising_steps", type=int, default=10, help="Diffusion denoising steps.")
    parser.add_argument("--domain_id", type=int, default=3, help="Domain id used by the model.")
    parser.add_argument(
        "--openvla_unnorm_key",
        type=str,
        default=None,
        help="De-normalization key for OpenVLA action stats when multiple datasets are present.",
    )
    parser.add_argument(
        "--gripper_close_threshold",
        type=float,
        default=0.5,
        help="Threshold for mapping gripper logits to open/close commands.",
    )
    parser.add_argument(
        "--save_video",
        dest="save_video",
        action="store_true",
        default=False,
        help="Save evaluation rollouts as videos (default: on).",
    )
    parser.add_argument(
        "--no_save_video",
        dest="save_video",
        action="store_false",
        help="Disable saving videos to speed up evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.task_suites is None:
        if args.sim == "libero":
            args.task_suites = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
        elif args.sim == "libero_plus":
            args.task_suites = ["libero_plus_camera", "libero_plus_robot", "libero_plus_language", "libero_plus_light", "libero_plus_background", "libero_plus_noise", "libero_plus_layout"]
        else:
            raise ValueError(f"Unknown sim type: {args.sim}")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    arch = _normalize_arch(args.model_arch)
    logger.info("Loading %s model on device %s", arch, device)
    model, processor = load_model_and_processor(args.model_path, args.processor_path, args.lora_path, device, arch)

    out_dir = Path(args.output_dir)
    logger.info(
        "Starting LIBERO evaluation | sim=%s | arch=%s | suites=%s | episodes=%s | save_dir=%s",
        args.sim,
        arch,
        args.task_suites,
        args.episodes,
        out_dir,
    )

    results = run_libero_eval(
        model=model,
        processor=processor,
        save_path=out_dir,
        task_suites=args.task_suites,
        num_episodes=args.episodes,
        init_seed=args.seed,
        act_type=args.act_type,
        denoising_steps=args.denoising_steps,
        domain_id=args.domain_id,
        device=device,
        save_video=args.save_video,
        model_arch=arch,
        openvla_unnorm_key=args.openvla_unnorm_key,
        gripper_close_threshold=args.gripper_close_threshold,
        sim_type=args.sim,
        max_tasks=args.max_tasks,
    )
    # Print detailed results
    sim_label = "LIBERO-Plus" if args.sim == "libero_plus" else "LIBERO"
    print("\n" + "=" * 80)
    print(f"📊 {sim_label} Evaluation Results Summary")
    print("=" * 80)
    for suite_name in args.task_suites:
        success_rate = results.get(suite_name, 0.0)
        successes = int(success_rate * args.episodes)
        avg_steps = results.avg_steps.get(suite_name, 0.0)
        print(
            f"  {suite_name:40s}: {success_rate:6.2%} ({successes}/{args.episodes}) | "
            f"avg steps: {avg_steps:7.2f}"
        )
    print("-" * 80)
    overall_avg = sum(results.values()) / len(results) if results else 0.0
    total_success = sum(int(sr * args.episodes) for sr in results.values())
    total_episodes = len(results) * args.episodes
    overall_avg_steps = (
        sum(results.avg_steps.get(suite, 0.0) * args.episodes for suite in results.avg_steps)
        / total_episodes
        if total_episodes
        else 0.0
    )
    print(
        f"  {'Overall Average':40s}: {overall_avg:6.2%} ({total_success}/{total_episodes}) | "
        f"avg steps: {overall_avg_steps:7.2f}"
    )
    print("=" * 80)
    logger.info("Evaluation finished: success=%s avg_steps=%s", dict(results), results.avg_steps)


if __name__ == "__main__":
    main()
