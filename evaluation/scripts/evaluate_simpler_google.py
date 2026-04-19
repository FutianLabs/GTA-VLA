#!/usr/bin/env python3
import argparse
import json
import logging
import os
import itertools
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sapien.core import Pose
from scipy.spatial.transform import Rotation as R
from transforms3d.euler import euler2quat

os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("SAPIEN_RENDERER", "offscreen")

import simpler_env  # noqa: E402
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict  # noqa: E402
from simpler_env.utils.visualization import write_video  # noqa: E402

from models.modeling_xvla import XVLA  # noqa: E402
from models.processing_xvla import build_xvla_processor  # noqa: E402

logger = logging.getLogger("evaluate_simpler_google")

TASK_TO_CONFIG = {
    "coke_can": "coke_can.json",
    "move_near": "move_near.json",
    "open_close": "open_close.json",
    "place_in": "place_in.json",
}


@dataclass
class TaskProfile:
    chunk_size: int
    gripper_threshold: float


def task_profile(setting: str, task: str) -> TaskProfile:
    if setting == "vm":
        if task == "place_in":
            return TaskProfile(chunk_size=6, gripper_threshold=0.28)
        if task == "open_close":
            return TaskProfile(chunk_size=10, gripper_threshold=0.35)
        return TaskProfile(chunk_size=10, gripper_threshold=0.25)
    if task == "place_in":
        return TaskProfile(chunk_size=10, gripper_threshold=0.3)
    return TaskProfile(chunk_size=10, gripper_threshold=0.25)


def parse_range_tuple(value):
    if isinstance(value, (int, float)):
        return [value]
    return np.linspace(value[0], value[1], int(value[2])).tolist()


def generate_robot_init_quats(quat_center, rpy_range):
    r_range = parse_range_tuple(rpy_range[:3])
    p_range = parse_range_tuple(rpy_range[3:6])
    y_range = parse_range_tuple(rpy_range[6:])
    return [
        (Pose(q=euler2quat(r, p, y)) * Pose(q=quat_center)).q
        for r, p, y in itertools.product(r_range, p_range, y_range)
    ]


def rotate6d_to_euler_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)
    return R.from_matrix(rot_mats).as_euler("xyz")


def rotate6d_to_matrix(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6, dtype=np.float32)
    a1 = v6[..., 0:3]
    a2 = v6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)


def sapien_quat_to_rot6d(q_wxyz: np.ndarray) -> np.ndarray:
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)
    mat = R.from_quat(q_xyzw).as_matrix().astype(np.float32)
    return mat[:, :2].reshape(6)


def obs_to_proprio_world(obs: dict) -> np.ndarray:
    tcp_pose = obs.get("extra", {}).get("tcp_pose", None)
    eef_pose = obs.get("agent", {}).get("eef_pos", None)
    if tcp_pose is not None and len(tcp_pose) >= 7:
        tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
        xyz = tcp_pose[:3]
        rot6d = sapien_quat_to_rot6d(tcp_pose[3:7])
        if eef_pose is not None and len(eef_pose) >= 8:
            eef_pose = np.asarray(eef_pose, dtype=np.float32)
            gripper = np.array([1.0 - eef_pose[7]], dtype=np.float32)
        else:
            gripper = np.zeros(1, dtype=np.float32)
    elif eef_pose is not None and len(eef_pose) >= 8:
        eef_pose = np.asarray(eef_pose, dtype=np.float32)
        xyz = eef_pose[:3]
        rot6d = sapien_quat_to_rot6d(eef_pose[3:7])
        gripper = np.array([1.0 - eef_pose[7]], dtype=np.float32)
    else:
        ee_pose_wrt_base = Pose(
            p=obs["agent"]["base_pose"][:3],
            q=obs["agent"]["base_pose"][3:],
        ).inv() * Pose(
            p=obs["extra"]["tcp_pose"][:3],
            q=obs["extra"]["tcp_pose"][3:],
        )
        xyz = np.asarray(ee_pose_wrt_base.p, dtype=np.float32)
        rot6d = sapien_quat_to_rot6d(np.asarray(ee_pose_wrt_base.q, dtype=np.float32))
        gripper = np.zeros(1, dtype=np.float32)
    left = np.concatenate([xyz, rot6d, gripper])
    out = np.zeros(20, dtype=np.float32)
    out[:10] = left
    return out


def obs_to_proprio_base(obs: dict) -> np.ndarray:
    ee_pose_wrt_base = Pose(
        p=obs["agent"]["base_pose"][:3],
        q=obs["agent"]["base_pose"][3:],
    ).inv() * Pose(
        p=obs["extra"]["tcp_pose"][:3],
        q=obs["extra"]["tcp_pose"][3:],
    )
    xyz = np.asarray(ee_pose_wrt_base.p, dtype=np.float32)
    rot6d = sapien_quat_to_rot6d(np.asarray(ee_pose_wrt_base.q, dtype=np.float32))
    eef_pose = obs.get("agent", {}).get("eef_pos", None)
    if eef_pose is not None and len(eef_pose) >= 8:
        eef_pose = np.asarray(eef_pose, dtype=np.float32)
        gripper = np.array([1.0 - eef_pose[7]], dtype=np.float32)
    else:
        gripper = np.zeros(1, dtype=np.float32)
    left = np.concatenate([xyz, rot6d, gripper])
    out = np.zeros(20, dtype=np.float32)
    out[:10] = left
    return out


def obs_to_proprio(obs: dict, frame: str = "world") -> np.ndarray:
    if frame == "base":
        return obs_to_proprio_base(obs)
    return obs_to_proprio_world(obs)


def apply_env_placeholders(obj, simpler_dir: str):
    if isinstance(obj, dict):
        return {k: apply_env_placeholders(v, simpler_dir) for k, v in obj.items()}
    if isinstance(obj, list):
        return [apply_env_placeholders(v, simpler_dir) for v in obj]
    if isinstance(obj, str):
        return obj.replace("{SIMPLER_DIR}", simpler_dir)
    return obj


def resolve_simpler_dir(simpler_dir: str) -> str:
    if simpler_dir:
        candidate = Path(simpler_dir).expanduser().resolve()
    else:
        candidate = Path(simpler_env.__file__).resolve().parents[1]

    if not candidate.is_dir():
        raise FileNotFoundError(
            f"Resolved SimplerEnv root does not exist: {candidate}. "
            "Pass --simpler_dir or set SIMPLER_DIR explicitly."
        )

    return str(candidate)


def load_xvla_model_and_processor(
    model_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Any]:
    model = XVLA.from_pretrained(model_path)
    vlm_type = getattr(model.config, "vlm_backbone_type", "qwen3_vl")
    if vlm_type != "qwen3_vl":
        raise ValueError(f"evaluate_simpler_google only supports qwen3_vl now, got: {vlm_type}")

    qwen3_path = "Qwen/Qwen3-VL-2B-Instruct"
    use_cot_training = getattr(model.config, "use_cot_training", False)
    num_views = int(getattr(model.config, "num_views", 1))
    logger.info("Qwen3 processor source: %s", qwen3_path)
    processor = build_xvla_processor(
        vlm_backbone_type="qwen3_vl",
        pretrained_name_or_path=qwen3_path,
        num_views=num_views,
        use_cot_training=use_cot_training,
        trust_remote_code=True,
    )

    uses_flash_attn = vlm_type == "qwen3_vl" and getattr(model.config, "qwen3_use_flash_attn", False)
    if uses_flash_attn:
        model = model.to(device).to(torch.bfloat16)
    else:
        model = model.to(device).to(torch.float32)
    model.eval()
    return model, processor


class LocalPolicyAdapter:
    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        device: torch.device,
        denoising_steps: int,
        domain_id: int,
        proprio_frame: str = "world",
        first_action_only: bool = False,
        xyz_bias_correction: np.ndarray | None = None,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.denoising_steps = denoising_steps
        self.domain_id = torch.tensor([domain_id], device=device, dtype=torch.long)
        self.dtype = next(model.parameters()).dtype
        self.proprio_frame = proprio_frame
        self.action_plan = deque()
        self.proprio = np.zeros(20, dtype=np.float32)
        self.instruction = ""
        self.profile = TaskProfile(chunk_size=10, gripper_threshold=0.25)
        self.first_action_only = bool(first_action_only)
        self.xyz_bias_correction = np.asarray(
            xyz_bias_correction if xyz_bias_correction is not None else [0.0, 0.0, 0.0],
            dtype=np.float32,
        ).reshape(3)

    def reset(self, proprio: np.ndarray, instruction: str, profile: TaskProfile):
        self.proprio = proprio.astype(np.float32)
        self.instruction = instruction
        self.profile = profile
        self.action_plan.clear()

    def update_proprio_from_obs(self, obs: dict):
        self.proprio[:] = obs_to_proprio(obs, frame=self.proprio_frame)

    def set_instruction(self, instruction: str):
        self.instruction = instruction

    def _prepare_inputs(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        pil_img = Image.fromarray(image)
        processed = self.processor(images=[pil_img], language_instruction=self.instruction)

        def to_device(t):
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            if t.is_floating_point():
                return t.to(device=self.device, dtype=self.dtype)
            return t.to(device=self.device)

        out = {k: to_device(v) for k, v in processed.items()}
        out["proprio"] = torch.as_tensor(self.proprio, device=self.device, dtype=self.dtype).unsqueeze(0)
        out["domain_id"] = self.domain_id
        return out

    def step(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if not self.action_plan:
            with torch.inference_mode():
                raw_action = (
                    self.model.generate_actions(steps=self.denoising_steps, **self._prepare_inputs(image))
                    .squeeze(0)
                    .float()
                    .cpu()
                    .numpy()
                )
            action_seq = np.asarray(raw_action, dtype=np.float32)
            if action_seq.ndim == 1:
                action_seq = action_seq[None, :]
            action_seq = action_seq[::2][: self.profile.chunk_size]
            action_seq = action_seq.copy()
            action_seq[:, :3] += self.xyz_bias_correction.reshape(1, 3)
            if self.first_action_only:
                action_seq = action_seq[:1]
            self.action_plan.extend(action_seq.tolist())

        action_pred = np.array(self.action_plan.popleft(), dtype=np.float32)
        curr_pose10 = self.proprio[:10].copy()
        curr_xyz = curr_pose10[:3]
        curr_rot6d = curr_pose10[3:9]
        target_xyz = action_pred[:3]
        target_rot6d = action_pred[3:9]
        delta_xyz = (target_xyz - curr_xyz).astype(np.float32)
        curr_rot = rotate6d_to_matrix(curr_rot6d)
        target_rot = rotate6d_to_matrix(target_rot6d)
        delta_rot = R.from_matrix(target_rot @ curr_rot.T).as_rotvec().astype(np.float32)
        action_final = np.concatenate(
            [
                delta_xyz,
                delta_rot,
                np.array([1.0 if action_pred[9] > self.profile.gripper_threshold else -1.0], dtype=np.float32),
            ]
        )
        raw_pred_pose10 = action_pred[:10].copy()
        raw_pred_pose10[:3] = raw_pred_pose10[:3] - self.xyz_bias_correction
        pred_pose10 = action_pred[:10].copy()
        debug_info: Dict[str, np.ndarray] = {
            "curr_pose10": curr_pose10,
            "raw_pred_pose10": raw_pred_pose10,
            "pred_pose10": pred_pose10,
            "env_action7": action_final.copy(),
        }
        return action_final, debug_info


def load_task_configs(google_setting: str, task: str, simpler_dir: str) -> Dict[str, dict]:
    base_name = "google-VM" if google_setting == "vm" else "google-VA"
    root = Path(__file__).resolve().parent.parent / "simpler" / base_name / "configs"
    cfg_path = root / TASK_TO_CONFIG[task]
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return apply_env_placeholders(raw, simpler_dir)


def build_reset_combinations(cfg: dict) -> List[dict]:
    combinations = []
    if cfg["obj_variation_mode"] == "episode":
        for ep_id in range(cfg["episode_nums"]):
            combinations.append({"episode_id": ep_id})
    elif cfg["obj_variation_mode"] == "xy":
        x_list = parse_range_tuple(cfg["obj_init_x_range"])
        y_list = parse_range_tuple(cfg["obj_init_y_range"])
        for x, y in itertools.product(x_list, y_list):
            combinations.append({"init_xy": np.array([x, y])})
    return combinations


def evaluate_task(
    task: str,
    google_setting: str,
    policy: LocalPolicyAdapter,
    output_dir: Path,
    save_video: bool,
    simpler_dir: str,
) -> Tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"google_results_{google_setting}.txt"
    summary_path = output_dir / f"google_summary_{google_setting}.txt"
    scenarios = load_task_configs(google_setting, task, simpler_dir)
    profile = task_profile(google_setting, task)

    total_episodes = 0
    success_count = 0

    for scenario_name, cfg in scenarios.items():
        scenario_success = 0
        scenario_episodes = 0
        max_steps = int(cfg["max_episode_steps"]) * 2
        robot_quats = generate_robot_init_quats(cfg["robot_init_rot_quat_center"], cfg["robot_init_rot_rpy_range"])
        robot_x_list = parse_range_tuple(cfg["robot_init_x"])
        robot_y_list = parse_range_tuple(cfg["robot_init_y"])

        for robot_init_x in robot_x_list:
            for robot_init_y in robot_y_list:
                for robot_init_quat in robot_quats:
                    make_kwargs = dict(
                        robot=cfg["robot_name"],
                        sim_freq=513,
                        control_freq=3,
                        control_mode="arm_pd_ee_base_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner",
                        scene_name=cfg["scene_name"],
                        camera_cfgs={"add_segmentation": True},
                        rgb_overlay_path=cfg.get("rgb_overlay_path", None),
                        rgb_overlay_cameras=cfg.get("rgb_overlay_cameras", None),
                    )
                    if "rgb_overlay_path" in cfg and "rgb_overlay_cameras" not in cfg and "google_robot_static" in cfg["robot_name"]:
                        make_kwargs["rgb_overlay_cameras"] = ["overhead_camera"]

                    env = simpler_env.make(cfg["env_name"], **make_kwargs, **cfg["additional_env_build_kwargs"])
                    options = {
                        "robot_init_options": {
                            "init_xy": np.array([robot_init_x, robot_init_y]),
                            "init_rot_quat": robot_init_quat,
                        }
                    }
                    try:
                        for obj_reset_option in build_reset_combinations(cfg):
                            options["obj_init_options"] = obj_reset_option
                            obs, _ = env.reset(options=options)
                            instruction = env.get_language_instruction()
                            policy.reset(obs_to_proprio(obs), instruction, profile)

                            done = False
                            reward = 0.0
                            frames: List[np.ndarray] = []
                            wall_start = time.time()
                            steps_taken = max_steps

                            for step_idx in range(max_steps):
                                policy.update_proprio_from_obs(obs)
                                instruction = env.get_language_instruction()
                                if instruction != policy.instruction:
                                    policy.set_instruction(instruction)
                                image = get_image_from_maniskill2_obs_dict(env, obs)
                                action, _ = policy.step(image)
                                obs, reward, done, _, _ = env.step(action)
                                if save_video:
                                    frames.append(image.copy())
                                if done:
                                    steps_taken = step_idx + 1
                                    break

                            duration = time.time() - wall_start
                            ep_id = total_episodes
                            video_path = output_dir / f"{scenario_name}_{ep_id}_{float(done):.2f}.mp4"
                            if save_video:
                                write_video(str(video_path), frames, fps=10)

                            row = {
                                "task": task,
                                "scenario": scenario_name,
                                "episode_id": ep_id,
                                "reward": float(reward),
                                "done": bool(done),
                                "steps": int(steps_taken),
                                "duration_sec": float(duration),
                                "output": str(video_path) if save_video else "",
                            }
                            with results_path.open("a", encoding="utf-8") as f:
                                f.write(json.dumps(row) + "\n")

                            total_episodes += 1
                            scenario_episodes += 1
                            if done:
                                success_count += 1
                                scenario_success += 1
                    finally:
                        env.close()

        summary = {
            "task": task,
            "scenario": scenario_name,
            "total_episodes": scenario_episodes,
            "success_count": scenario_success,
            "success_rate": (scenario_success / scenario_episodes) if scenario_episodes else 0.0,
        }
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

    return success_count, total_episodes


def parse_args():
    parser = argparse.ArgumentParser("Simpler Google evaluation (vm/va)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--processor_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--google_setting", type=str, default="vm", choices=["vm", "va"])
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["coke_can", "move_near", "open_close", "place_in"],
        choices=["coke_can", "move_near", "open_close", "place_in"],
    )
    parser.add_argument("--output_dir", type=str, default="evaluation_outputs/simpler_google")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--denoising_steps", type=int, default=10)
    parser.add_argument("--domain_id", type=int, default=1)
    parser.add_argument("--proprio_frame", type=str, default="world", choices=["world", "base"])
    parser.add_argument("--first_action_only", action="store_true",
                        help="Replan every environment step and execute only the first action from each predicted chunk")
    parser.add_argument("--xyz_bias_correction", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Additive xyz correction applied to predicted absolute pose before delta conversion")
    parser.add_argument("--save_video", dest="save_video", action="store_true", default=False)
    parser.add_argument("--no_save_video", dest="save_video", action="store_false")
    parser.add_argument("--simpler_dir", type=str, default=os.getenv("SIMPLER_DIR", ""))
    args = parser.parse_args()
    args.simpler_dir = resolve_simpler_dir(args.simpler_dir)
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, processor = load_xvla_model_and_processor(
        model_path=args.model_path,
        device=device,
    )
    policy = LocalPolicyAdapter(
        model=model,
        processor=processor,
        device=device,
        denoising_steps=args.denoising_steps,
        domain_id=args.domain_id,
        proprio_frame=args.proprio_frame,
        first_action_only=args.first_action_only,
        xyz_bias_correction=np.asarray(args.xyz_bias_correction, dtype=np.float32),
    )

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    task_rates = {}
    total_success = 0
    total_episodes = 0

    for task in args.tasks:
        task_dir = out_root / f"{args.google_setting}_{task}"
        success, episodes = evaluate_task(
            task=task,
            google_setting=args.google_setting,
            policy=policy,
            output_dir=task_dir,
            save_video=args.save_video,
            simpler_dir=args.simpler_dir,
        )
        rate = (success / episodes) if episodes else 0.0
        task_rates[task] = rate
        total_success += success
        total_episodes += episodes

    overall_avg = sum(task_rates.values()) / len(task_rates) if task_rates else 0.0

    summary_json = {
        "google_setting": args.google_setting,
        "tasks": task_rates,
        "overall_avg": overall_avg,
        "total_success": total_success,
        "total_episodes": total_episodes,
    }
    (out_root / f"all_tasks_summary_{args.google_setting}.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("Simpler Google Evaluation Results Summary")
    print("=" * 80)
    for task in args.tasks:
        rate = task_rates[task]
        print(f"  {args.google_setting}_{task:13s} : {rate * 100:6.2f}%")
    print("-" * 80)
    print(f"  {'Overall Average':16s} : {overall_avg * 100:6.2f}% ({total_success}/{total_episodes})")
    print("=" * 80)


if __name__ == "__main__":
    main()
