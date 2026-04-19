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
from typing import Dict, Iterable, List, Optional, Any
from pathlib import Path
import io, json, os, random, numpy as np, torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from mmengine import fileio
from .utils import action_slice
from .domain_config import DATA_WEIGHTS, DATA_DOMAIN_ID
from .domain_handler.registry import get_handler_cls

# CoT handler mapping: base handler name -> CoT handler name
# Only applied when use_cot=True AND meta has annotation_dir
_COT_HANDLER_MAP = {
    "Bridge": "BridgeCot",
    "Fractal": "FractalCot",
    "Droid-Left": "DroidCot-Left",
    "Droid-Right": "DroidCot-Right",
    "robomind-franka": "robomind-franka-cot",
    "robomind-franka-1rgb": "robomind-franka-cot",
    "robomind-franka-3rgb": "robomind-franka-cot",
    "robomind-ur": "robomind-ur-cot",
}

class InfiniteDataReader(IterableDataset):
    """
    Unified VLA dataset reader supporting different VLM backbones.
    
    Output sample format depends on image_processor:
      - image_processor=None (Florence2):
        'image_input': FloatTensor[V, C, H, W]  # Raw normalized pixels
        
      - image_processor provided (Qwen3-VL):
        'image_input': FloatTensor[V, num_patches, embed_dim]  # Pre-embedded patches
        'image_grid_thw': LongTensor[V, 3]  # Grid metadata
    
    Common fields:
      {
        'domain_id': LongTensor[],
        'language_instruction': str,
        'image_mask': BoolTensor[V],
        'proprio': FloatTensor[dim_proprio],
        'action': FloatTensor[T, dim_action]
      }
    """
    def __init__(self, 
                 metas_path: str, 
                 num_actions: int = 10, 
                 training: bool = True,
                 action_mode: str = "ee6d",
                 lang_aug: str = None,
                 image_color_jitter: bool = True,
                 image_processor = None,  # VLM-specific image processor
                 model_config: Any = None,  # XVLAConfig for CoT support
                 ):
        # Read num_views from model_config (default 3)
        if model_config is not None:
            self.num_views = getattr(model_config, 'num_views', 3) if not isinstance(model_config, dict) else model_config.get('num_views', 3)
        else:
            self.num_views = 3
        self.training = training
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.image_processor = image_processor
        self.metas: Dict[str, dict] = {}
        self.model_config = model_config
        
        # Check if CoT training is enabled from model config
        self.use_cot = False
        self.aug_with_optional_view = False
        self.use_dual_frequency = False
        self.vlm_frame_max_offset = 10
        if model_config is not None:
            if hasattr(model_config, 'use_cot_training'):
                self.use_cot = model_config.use_cot_training
            elif isinstance(model_config, dict):
                self.use_cot = model_config.get('use_cot_training', False)
            if hasattr(model_config, 'aug_with_optional_view'):
                self.aug_with_optional_view = model_config.aug_with_optional_view
            elif isinstance(model_config, dict):
                self.aug_with_optional_view = model_config.get('aug_with_optional_view', False)
            if hasattr(model_config, 'use_dual_frequency'):
                self.use_dual_frequency = model_config.use_dual_frequency
            elif isinstance(model_config, dict):
                self.use_dual_frequency = model_config.get('use_dual_frequency', False)
            if hasattr(model_config, 'vlm_frame_max_offset'):
                self.vlm_frame_max_offset = model_config.vlm_frame_max_offset
            elif isinstance(model_config, dict):
                self.vlm_frame_max_offset = model_config.get('vlm_frame_max_offset', 10)
        
        print("use action mode:", action_mode)
        if self.use_cot:
            print("🔗 CoT training enabled - will use CoT handlers where available")
        if self.use_dual_frequency:
            print(f"🔀 Dual-frequency training enabled - vlm_frame_max_offset={self.vlm_frame_max_offset}")
        
        if fileio.isdir(metas_path):
            meta_files = fileio.list_dir_or_file(metas_path, suffix=".json", recursive=True, list_dir=False)
            root = metas_path
        else: meta_files, root = [metas_path], ""
        for file in meta_files:
            with io.BytesIO(fileio.get(fileio.join_path(root, file))) as f: meta = json.load(f)
            print(f"== dataset {meta['dataset_name']}")
            self.metas[meta["dataset_name"]] = meta

        self.export_align_dir = os.getenv("XVLA_EXPORT_ALIGN_DIR", "").strip()
        self.export_align_limit = int(os.getenv("XVLA_EXPORT_ALIGN_LIMIT", "0"))
        self.export_align_dataset = os.getenv("XVLA_EXPORT_ALIGN_DATASET", "").strip()
        self._export_align_count = 0
        self._export_align_writer = None
        if self.export_align_dir:
            export_root = Path(self.export_align_dir).expanduser().resolve()
            (export_root / "images").mkdir(parents=True, exist_ok=True)
            self._export_align_writer = (export_root / "samples.jsonl").open("w", encoding="utf-8")
            print(f"alignment export enabled: {export_root}")


        # Setup image augmentation
        use_jitter = (image_color_jitter == True or image_color_jitter == "True") and training
        color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.) if use_jitter else transforms.Lambda(lambda x: x)
        
        if image_processor is not None:
            # VLM processor: only color jitter, processor handles rest
            self.image_aug = None
        else:
            # Default: ImageNet preprocessing
            self.image_aug = transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                color_jitter,
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
            ])
            print("Using ImageNet preprocessing" + (" + color jitter" if use_jitter else ""))

    def _export_alignment_sample(self, dataset_name: str, sample: dict):
        if self._export_align_writer is None:
            return
        if self.export_align_dataset and dataset_name != self.export_align_dataset:
            return
        if self.export_align_limit > 0 and self._export_align_count >= self.export_align_limit:
            return

        raw_images = sample.get("raw_images")
        abs_traj = sample.get("abs_trajectory")
        if raw_images is None or abs_traj is None:
            return
        if not isinstance(abs_traj, torch.Tensor) or abs_traj.ndim != 2 or abs_traj.shape[0] < 2:
            return

        self._export_align_count += 1
        sample_id = f"{dataset_name}_traj{int(sample.get('traj_idx', -1)):06d}_sample{int(sample.get('sample_idx', -1)):06d}"
        export_root = Path(self.export_align_dir).expanduser().resolve()
        image_rel = Path("images") / f"{sample_id}.png"
        image_path = export_root / image_rel

        image_tensor = raw_images[0].detach().cpu()
        image_np = image_tensor.permute(1, 2, 0).numpy()
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
        transforms.ToPILImage()(torch.from_numpy(image_np).permute(2, 0, 1)).save(image_path)

        next_proprio = abs_traj[1].detach().cpu().numpy().astype(np.float32).tolist()
        cur_proprio = abs_traj[0].detach().cpu().numpy().astype(np.float32).tolist()
        record = {
            "sample_id": sample_id,
            "dataset_name": dataset_name,
            "traj_idx": int(sample.get("traj_idx", -1)),
            "sample_idx": int(sample.get("sample_idx", -1)),
            "frame_idx": int(sample.get("frame_idx", -1)),
            "language_instruction": sample.get("language_instruction", ""),
            "image_path": str(image_path),
            "current_proprio": cur_proprio,
            "next_proprio": next_proprio,
        }
        self._export_align_writer.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._export_align_writer.flush()

    def __del__(self):
        writer = getattr(self, "_export_align_writer", None)
        if writer is not None:
            writer.close()

    def _iter_one_dataset(self, dataset_name: str) -> Iterable[dict]:
        meta = self.metas[dataset_name]
        
        # Determine handler: use CoT handler if enabled and mapped
        handler_name = dataset_name
        if self.use_cot and dataset_name in _COT_HANDLER_MAP:
            if not meta.get("annotation_dir"):
                raise ValueError(
                    f"CoT training enabled for '{dataset_name}' but meta has no 'annotation_dir'. "
                    f"Add annotation_dir to the meta JSON or disable CoT training."
                )
            handler_name = _COT_HANDLER_MAP[dataset_name]
            print(f"📝 Using CoT handler: {handler_name} for {dataset_name}")
        
        Handler = get_handler_cls(handler_name)
        handler = Handler(meta=meta, num_views=self.num_views)
        handler.aug_with_optional_view = self.aug_with_optional_view
        handler.use_dual_frequency = self.use_dual_frequency
        handler.vlm_frame_max_offset = self.vlm_frame_max_offset
        
        # If CoT handler, set config from model config
        if self.use_cot and hasattr(handler, 'set_cot_config') and self.model_config is not None:
            handler.set_cot_config(self.model_config)
        
        # If interaction augmentation is enabled, set config
        if hasattr(handler, 'set_interaction_config') and self.model_config is not None:
            handler.set_interaction_config(self.model_config)
        
        traj_indices = list(range(len(meta["datalist"])))
        if self.training: random.shuffle(traj_indices)
        print("traj_indices:", len(traj_indices))

        for traj_idx in traj_indices:
            try:
                
                for sample_idx, sample in enumerate(handler.iter_episode(
                    traj_idx,
                    num_actions=self.num_actions,
                    training=self.training,
                    image_aug=self.image_aug,
                    lang_aug_map= meta["lang_aug_map"] if "lang_aug_map" in meta.keys() else None,
                    action_mode = self.action_mode,
                    image_processor=self.image_processor
                )):
                    # print("sample keys:", sample.keys())
                    if self.use_cot and not sample.get("cot_text", "").strip():
                        datapath = meta["datalist"][traj_idx]
                        if not isinstance(datapath, str):
                            datapath = datapath[0]
                        print(f"⚠️ [CoT skip] empty cot_text | dataset={dataset_name} | traj_idx={traj_idx} | sample_idx={sample_idx} | path={datapath}")
                        continue
                    sample["domain_id"] = torch.tensor(DATA_DOMAIN_ID.get(dataset_name, 0))
                    sample["sample_idx"] = sample_idx
                    sample["traj_idx"] = traj_idx
                    self._export_alignment_sample(dataset_name, sample)
                    sample.update(action_slice(sample["abs_trajectory"],
                                  action_mode=self.action_mode))
                    del sample["abs_trajectory"]
                    yield sample
            except Exception as e:
                print("reading error, traj_idx:", traj_idx, meta["datalist"][traj_idx])
                import traceback; traceback.print_exc()
                with open("error_log.txt", "a") as f: f.write(f"skip broken traj {meta['datalist'][traj_idx]} with {e}\n")
                continue
        if self.training: yield from self._iter_one_dataset(dataset_name)


    def __iter__(self):
        names = list(self.metas.keys())
        if not self.training: 
            for n in names: yield from self._iter_one_dataset(n)
        else:
            #names = names * 2 # increase the dataset sampling frequency
            gens = [iter(self._iter_one_dataset(n)) for n in names]
            ws = [DATA_WEIGHTS.get(n, 1.0) for n in names]
            s = sum(ws); ws = [w / s for w in ws]
            
            # Print dataset weights
            print("\n" + "="*60)
            print("📊 Multi-Dataset Training Weights:")
            print("="*60)
            for name, weight in zip(names, ws):
                num_trajs = len(self.metas[name]["datalist"])
                print(f"  {name:30s} | Weight: {weight:6.2%} | Trajs: {num_trajs:6d}")
            print("="*60 + "\n")
            
            while True:
                i = random.choices(range(len(names)), weights=ws, k=1)[0]
                yield next(gens[i])
