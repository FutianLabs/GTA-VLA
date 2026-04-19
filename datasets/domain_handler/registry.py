# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Type
from .base import DomainHandler

# Handlers
from .lerobot_agibot import AGIBOTLeRobotHandler
from .agiworld import AGIWolrdHandler
from .robomind import RobomindHandler
from .droid import DroidHandler
from .real_world import AIRAgilexHandler, AIRAgilexHQHandler, AIRBotHandler, WidowxAirHandler
from .piper import PiperHandler
from .piper_joint import PiperJointHandler
from .simulations import BridgeHandler, LiberoHandler, VLABenchHandler, RobotWin2Handler, RobocasaHumanHandler, CalvinHandler, RT1Handler, FractalHandler
from .simulations_cot import BridgeCotHandler, DroidCotHandler, RobomindCotHandler, FractalCotHandler, ManiSkillCotHandler
# 1) Exact registry only (no heuristics)
_REGISTRY: Dict[str, Type[DomainHandler]] = {
    # LeRobot (parquet)
    "AGIBOT": AGIBOTLeRobotHandler,
    "AGIBOT-challenge": AGIBOTLeRobotHandler,

    # HDF5 (exact)
    "Calvin": CalvinHandler,
    "RT1": RT1Handler,
    "Fractal": FractalHandler,

    # AIR family
    "AIR-AGILEX": AIRAgilexHandler,
    "AIR-AGILEX-HQ": AIRAgilexHQHandler,
    "AIRBOT": AIRBotHandler,
    "widowx-air": WidowxAirHandler,

    # Sim/others
    "Bridge": BridgeHandler,
    "BridgeCot": BridgeCotHandler,  # Bridge with CoT annotations
    "ManiSkillCot": ManiSkillCotHandler,
    "FractalCot": FractalCotHandler,
    "libero": LiberoHandler,
    "VLABench": VLABenchHandler,
    "robotwin2_abs_ee": RobotWin2Handler,
    "robotwin2_clean": RobotWin2Handler,
    "robocasa-human": RobocasaHumanHandler,

    # Robomind
    "robomind-franka": RobomindHandler,
    "robomind-franka-3rgb": RobomindHandler,
    "robomind-franka-1rgb": RobomindHandler,
    "robomind-ur": RobomindHandler,
    "robomind-agilex": RobomindHandler,
    "robomind-franka-dual": RobomindHandler,

    # Droid
    "Droid-Left": DroidHandler,
    "Droid-Right": DroidHandler,
    "DroidCot-Left": DroidCotHandler,
    "DroidCot-Right": DroidCotHandler,

    # RoboMIND CoT
    "robomind-franka-cot": RobomindCotHandler,
    "robomind-ur-cot": RobomindCotHandler,
    
    
    "agiworld-on-site-pack": AGIWolrdHandler ,
    "agiworld-on-site-pack-extra": AGIWolrdHandler ,
    "agiworld-on-site-conveyor": AGIWolrdHandler ,
    "agiworld-on-site-conveyor-extra": AGIWolrdHandler ,
    "agiworld-on-site-restock": AGIWolrdHandler ,
    "agiworld-on-site-pour": AGIWolrdHandler ,
    "agiworld-on-site-microwave": AGIWolrdHandler ,
    "agiworld-on-site-cloth": AGIWolrdHandler,
    "agiworld-on-site-cloth-2": AGIWolrdHandler,

    # Piper
    "piper": PiperHandler,
    "piper_joint": PiperJointHandler,
}

def get_handler_cls(dataset_name: str) -> Type[DomainHandler]:
    """Strict lookup: require explicit registration."""
    try:
        return _REGISTRY[dataset_name]
    except KeyError:
        raise KeyError(
            f"No handler registered for dataset '{dataset_name}'. "
            f"Add it to _REGISTRY in datasets/domain_handler/registry.py."
        )


