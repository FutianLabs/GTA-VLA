from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Iterable

import numpy as np
import h5py

from ..utils import euler_to_rotate6d
from .base import BaseHDF5Handler


class DroidHandler(BaseHDF5Handler):

    dataset_name = "Droid-*"

    def build_left_right(
        self, proprio: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 15.0, 4.0
        left = np.concatenate(
            [proprio[:, :3], euler_to_rotate6d(proprio[:, 3:6], "xyz"), action[:, 6:7]],
            axis=-1,
        )
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 30))
