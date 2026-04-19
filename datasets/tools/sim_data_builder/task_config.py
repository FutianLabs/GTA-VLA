"""
Task mapping configuration for Bridge-aligned WidowX simulation data collection.

Maps task_key -> (env_id, task_group, default_episodes, max_steps, scene_name, use_wrist).
Covers three task groups: base, multi_object, layout_distractor.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TaskConfig:
    task_key: str
    env_id: str
    task_group: str
    default_episodes: int
    max_steps: int = 120
    robot: str = "widowx"
    scene_name: str = "bridge_table_1_v1"
    use_wrist: bool = True
    extra_env_kwargs: Dict = field(default_factory=dict)


# ── Base Tasks ──────────────────────────────────────────────────────────────
_BASE_TASKS = [
    TaskConfig(
        task_key="widowx_carrot_on_plate",
        env_id="PutCarrotOnPlateInScene-v0",
        task_group="base",
        default_episodes=24,
    ),
    TaskConfig(
        task_key="widowx_spoon_on_towel",
        env_id="PutSpoonOnTableClothInScene-v0",
        task_group="base",
        default_episodes=24,
    ),
    TaskConfig(
        task_key="widowx_stack_cube",
        env_id="StackGreenCubeOnYellowCubeInScene-v0",
        task_group="base",
        default_episodes=24,
    ),
    TaskConfig(
        task_key="widowx_put_bridge_objects_on_plate",
        env_id="PutBridgeObjectsOnPlateMultiObjectInScene-v0",
        task_group="base",
        default_episodes=60,
    ),
]

_BASIC3_TASK_KEYS = [
    "widowx_carrot_on_plate",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
]

# ── Multi-Object Tasks ──────────────────────────────────────────────────────
_MULTI_OBJECT_TASKS = [
    TaskConfig(
        task_key="widowx_carrot_on_plate_multi_object",
        env_id="PutCarrotOnPlateMultiObjectInScene-v0",
        task_group="multi_object",
        default_episodes=60,
    ),
    TaskConfig(
        task_key="widowx_spoon_on_towel_multi_object",
        env_id="PutSpoonOnTableClothMultiObjectInScene-v0",
        task_group="multi_object",
        default_episodes=60,
    ),
    TaskConfig(
        task_key="widowx_stack_cube_multi_object",
        env_id="StackCubeMultiObjectInScene-v0",
        task_group="multi_object",
        default_episodes=60,
    ),
]

# ── Layout-Distractor Tasks ─────────────────────────────────────────────────
_LAYOUT_DISTRACTOR_TASKS = [
    TaskConfig(
        task_key="widowx_carrot_on_plate_layout_distractor",
        env_id="PutCarrotOnPlateLayoutDistractorInScene-v0",
        task_group="layout_distractor",
        default_episodes=24,
    ),
    TaskConfig(
        task_key="widowx_spoon_on_towel_layout_distractor",
        env_id="PutSpoonOnTableClothLayoutDistractorInScene-v0",
        task_group="layout_distractor",
        default_episodes=24,
    ),
    TaskConfig(
        task_key="widowx_stack_cube_layout_distractor",
        env_id="StackGreenCubeOnYellowCubeLayoutDistractorInScene-v0",
        task_group="layout_distractor",
        default_episodes=24,
    ),
]

# ── Eggplant（仍可通过 --task 指定，不包含在 --group 批量运行）────────────────
_EGGPLANT_TASKS = [
    TaskConfig(
        task_key="widowx_put_eggplant_in_basket",
        env_id="PutEggplantInBasketScene-v0",
        task_group="base",
        default_episodes=24,
    ),
    TaskConfig(
        task_key="widowx_put_eggplant_in_basket_multi_object",
        env_id="PutEggplantInBasketMultiObjectInScene-v0",
        task_group="multi_object",
        default_episodes=60,
    ),
    TaskConfig(
        task_key="widowx_put_eggplant_in_basket_layout_distractor",
        env_id="PutEggplantInBasketLayoutDistractorInScene-v0",
        task_group="layout_distractor",
        default_episodes=24,
    ),
]

# ── Registry ────────────────────────────────────────────────────────────────
ALL_TASKS: List[TaskConfig] = (
    _BASE_TASKS + _MULTI_OBJECT_TASKS + _LAYOUT_DISTRACTOR_TASKS + _EGGPLANT_TASKS
)

TASK_MAP: Dict[str, TaskConfig] = {t.task_key: t for t in ALL_TASKS}

GROUP_MAP: Dict[str, List[TaskConfig]] = {}
for _t in _BASE_TASKS + _MULTI_OBJECT_TASKS + _LAYOUT_DISTRACTOR_TASKS:
    GROUP_MAP.setdefault(_t.task_group, []).append(_t)


def get_task(task_key: str) -> TaskConfig:
    if task_key not in TASK_MAP:
        raise KeyError(
            f"Unknown task_key '{task_key}'. "
            f"Available: {sorted(TASK_MAP.keys())}"
        )
    return TASK_MAP[task_key]


def get_group(group_name: str) -> List[TaskConfig]:
    if group_name not in GROUP_MAP:
        raise KeyError(
            f"Unknown group '{group_name}'. "
            f"Available: {sorted(GROUP_MAP.keys())}"
        )
    return GROUP_MAP[group_name]


def list_all_task_keys() -> List[str]:
    return [t.task_key for t in ALL_TASKS]


def list_group_run_task_keys() -> List[str]:
    """与 --group base/multi_object/layout_distractor 一致（不含单独挂起的 eggplant）。"""
    keys: List[str] = []
    for name in ("base", "multi_object", "layout_distractor"):
        keys.extend(t.task_key for t in get_group(name))
    return keys


def list_basic3_task_keys() -> List[str]:
    return list(_BASIC3_TASK_KEYS)
