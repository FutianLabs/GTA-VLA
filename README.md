# GTA-VLA

GTA-VLA (Guide, Think, Act) is an interactive Vision-Language-Action framework for spatially steerable embodied reasoning. It lets users optionally guide robot policies with visual cues such as affordance points, boxes, and traces, then conditions spatial-visual chain-of-thought reasoning and action generation on that guidance.

This repository provides the public code release for training, evaluation, metadata preparation, and model loading. Raw datasets, experiment logs, and local environment artifacts are intentionally excluded.

## Links

- Paper: [Guide, Think, Act: Interactive Embodied Reasoning in Vision-Language-Action Models](https://arxiv.org/abs/2605.13632)
- Project page: [GTA-VLA Project Page](https://signalispupupu.github.io/GTA-VLA_ProjPage/)
- Checkpoints: [SignalIsPuPuPu/GTA-VLA](https://huggingface.co/SignalIsPuPuPu/GTA-VLA)

The paper reports an 81.2% success rate on the SimplerEnv WidowX benchmark and shows that one-shot spatial guidance can improve recovery under visual shifts and spatial ambiguity.

## Status

This repository is the public GTA-VLA code release.

Current scope:

- Core model code
- Dataset loading and preprocessing logic
- Training and evaluation scripts
- Configuration files
- Metadata examples


## Repository Layout

```text
GTA-VLA/
├── models/               # Core model definitions
├── datasets/             # Dataset loading and preprocessing
├── configs/              # Training and evaluation configs
├── evaluation/           # Benchmark and evaluation entrypoints
├── scripts/              # Launch scripts for training and evaluation
├── tools/                # Metadata and preprocessing utilities
├── data/                 # Metadata examples only
├── train.py              # Main training entrypoint
├── peft_train.py         # PEFT training entrypoint
└── auto_eval_on_checkpoint.py
```

## Installation

GTA-VLA uses a **Python 3.10** environment managed with **[uv](https://github.com/astral-sh/uv)**.
Both [`.python-version`](.python-version) and [`pyproject.toml`](pyproject.toml) target Python 3.10.

### Prerequisites

- Linux with Python 3.10 installed
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- NVIDIA GPU and CUDA 12.x for GPU training or evaluation

### 1. Clone the repository

```bash
git clone <YOUR_GTA-VLA_REPO_URL>
cd GTA-VLA
```

### 2. Create and activate the uv environment

```bash
uv venv --python 3.10
source .venv/bin/activate
```

On Windows PowerShell, use:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install the Python package and core dependencies

For training, preprocessing, and code development, install GTA-VLA in editable mode and then install the repository requirements:

```bash
uv pip install -e .
uv pip install -r requirements.txt
```

The default dependency set includes PyTorch 2.8 and CUDA 12 runtime wheels. Before installation, make sure your NVIDIA driver is compatible with the CUDA 12.x stack used by your machine.

### 4. Handle optional local evaluation dependencies

[`requirements.txt`](requirements.txt) keeps SimplerEnv-related editable installs commented out by default. This lets the core GTA-VLA package install on machines that do not have local simulator checkouts.

If you need simulation-based evaluation, install the external environments first and then either install them manually or uncomment and update those editable dependency paths to match your local checkout.

Example setup for `SimplerEnv`:

```bash
git clone https://github.com/255isWhite/SimplerEnv.git --recurse-submodules
cd SimplerEnv/ManiSkill2_real2sim
uv pip install -e .
cd ..
uv pip install -e .
cd /path/to/GTA-VLA
```

After that, either:

- make sure the editable `-e ./third_party/...` paths in [`requirements.txt`](requirements.txt) match your local checkout layout, or
- keep those lines commented out if you have already installed the required packages manually.

### 5. Verify the installation

```bash
python -c "import torch; import transformers; import datasets; print('GTA-VLA environment is ready')"
```

If the command completes without errors, the core environment is ready.

## Data Preparation

The repository already includes metadata examples for several training sources, including Bridge, Droid, RoboMIND-style datasets, and AgiBot-style datasets. Additional benchmark-specific configs and launch scripts are provided for Fractal, LIBERO, ManiSkill, and related evaluation setups.

GTA-VLA reads data through metadata JSON files rather than assuming a single fixed dataset root. You can pass either a single metadata file or a directory containing multiple `.json` metadata files to `--train_metas_path`.

Each metadata file is expected to define at least the following fields:

- `dataset_name`: dataset identifier used to select the correct domain handler
- `datalist`: list of trajectory paths
- `observation_key`: one or more image observation keys to read from each trajectory
- `language_instruction_key`: field containing the language instruction

Optional fields used by some training recipes include:

- `annotation_dir`: required for CoT-style training when the selected dataset handler expects extra annotations
- `lang_aug_map`: optional instruction augmentation mapping

The raw trajectories themselves can live anywhere on disk as long as the paths recorded in `datalist` are valid for the corresponding dataset handler. The current metadata examples mostly point to per-episode HDF5 files, while some metadata-generation utilities can also produce episode-directory lists.

The repository includes helper scripts for building or filtering metadata, for example:

- `datasets/gen_hdf5_meta.py` for assembling metadata from dataset roots
- `tools/gen_droid_meta.py` and `tools/gen_robomind_meta.py` for dataset-specific metadata generation
- `tools/gen_cot_meta.py`, `tools/gen_cot_meta_filtered.py`, and `tools/gen_meta_filtered.py` for preparing filtered or annotation-augmented metadata

The files under `data/` are examples and templates only. They are useful for understanding the expected schema, but their embedded paths should be treated as placeholders and adjusted for your local dataset layout.

Important note:

- This repository does not ship raw datasets.
- Paths in local development scripts may still need to be replaced with public placeholders.

## Training

The recommended training entrypoint is the unified launcher in `scripts/train.sh`.

Minimal example:

```bash
bash scripts/train.sh data/bridge_meta.json bridge_exp1
```

This command infers the dataset prefix from the metadata filename, creates a timestamped output directory, and launches either `accelerate` or `deepspeed` depending on the number of visible GPUs.

Replace the metadata and config paths with your local files before launching.

To train from scratch instead of starting from a pretrained base model:

```bash
bash scripts/train.sh <run_name> [--scratch] \
--train_metas_path /path/to/maniskill_meta.json \
--config_path configs/<domain>/<config>.json
```

If you are not using `--scratch`, set the base model path before launching:

```bash
export GTA_VLA_BASE_MODEL=/path/to/base_model
bash scripts/train.sh data/bridge_meta.json bridge_exp1
```

The main training arguments are defined in `train.py`. In practice, the most important ones are:

- `--train_metas_path`: metadata file or directory of metadata files
- `--output_dir`: output directory for logs and checkpoints
- `--models`: pretrained model path or Hugging Face repository to initialize from
- `--config_path`: model/config JSON used for scratch training or recipe selection
- `--batch_size`, `--iters`, `--learning_rate`, `--grad_accum`: core optimization settings
- `--save_interval`, `--log_interval`: checkpoint and logging frequency
- `--model_arch`: model family selector

Output directories are created under `logs/gtavla/` by default. The unified launcher uses the metadata name to build a path of the form:

```text
logs/gtavla/<dataset>/<run_name>-MM-DD-HH-MM
```

Special cases used by the launcher:

- single-GPU local runs default to `logs/gtavla/<dataset>_dummy/...`
- scratch runs default to `logs/gtavla/<dataset>_scratch/...`

Inside each run directory you should expect:

- `train.log`: training log written by the Python entrypoint
- `wandb_run_id.txt`: saved Weights & Biases run id when tracking is enabled
- `ckpt-<step>/`: checkpoint directories saved every `save_interval` steps and at the final step
- `ckpt-<step>/state.json`: serialized training step metadata for resume/evaluation tooling

## Evaluation

For checkpoint-based evaluation, use the monitor script:

```bash
python auto_eval_on_checkpoint.py /path/to/run_dir 60000 libero --max_tasks 50
```

This command watches the specified run directory, discovers `ckpt-<step>` subdirectories, evaluates new checkpoints, and writes evaluation logs under each checkpoint directory.

For a one-shot LIBERO-Plus sweep over all seven perturbation types:

```bash
bash scripts/eval_libero_plus.sh /path/to/ckpt-60000
```

Optional external environments are required for several benchmarks:

- LIBERO and LIBERO-Plus require the corresponding simulator environment and benchmark assets
- Simpler-style evaluation depends on `SimplerEnv` and `ManiSkill2_real2sim`
- Custom real-world evaluation scripts may require connection metadata, robot-side services, or task-specific runtime assets

Benchmark dependencies in this repository are organized around the evaluation entrypoints:

- `auto_eval_on_checkpoint.py` handles checkpoint discovery, task scheduling, and summary aggregation
- `scripts/eval_libero.sh` and `scripts/eval_libero_plus.sh` cover LIBERO-style evaluation
- `scripts/eval_fractal.sh`, `scripts/eval_new_objects.sh`, and `scripts/eval_variant_tasks.sh` cover additional simulator-based evaluation flows
- `scripts/eval_customed_env.sh` is an example of a task launcher that depends on local connection information and an already prepared runtime environment

If you only need offline training, you do not need to install every evaluation backend. For reproducible benchmark runs, install the relevant simulator stack first and then verify that the paths and environment variables in the corresponding evaluation script match your machine.

## Checkpoints

Public checkpoints are hosted on Hugging Face at [SignalIsPuPuPu/GTA-VLA](https://huggingface.co/SignalIsPuPuPu/GTA-VLA). You can use that repository as `GTA_VLA_BASE_MODEL` or pass a downloaded checkpoint path to scripts that accept `--models`.

The repository currently expects checkpoints to live inside a run directory and to be named by training step:

```text
<run_dir>/
├── train.log
├── wandb_run_id.txt
├── ckpt-5000/
├── ckpt-10000/
└── ckpt-<final_step>/
```

Each checkpoint directory is created by the training entrypoint and is expected to contain model files saved with `save_pretrained(...)` together with training state metadata:

```text
ckpt-<step>/
├── config.json
├── model*.safetensors
├── state.json
└── eval/
	├── logs/
	│   ├── regular.log
	│   └── final.log
	└── summary/
		└── results_*.tsv
```

Notes:

- `state.json` stores the saved training step and is used by resume/evaluation tooling
- the `eval/` subtree is created only after running evaluation
- checkpoint discovery in `auto_eval_on_checkpoint.py` relies on the `ckpt-<step>` naming pattern

## Known Limitations

The current public export still has several limitations:

- optional simulation evaluation dependencies in `requirements.txt` are documented as commented local editable installs and must be installed separately when needed
- metadata examples under `data/` contain development-time dataset paths and are not drop-in runnable without path updates
- some evaluation flows assume locally prepared simulator checkouts, benchmark assets, or connection files
- end-to-end installation and benchmark CI validation are not included in this release snapshot

## Citation

If you use GTA-VLA, please cite:

```bibtex
@article{ling2026guidethinkact,
  title={Guide, Think, Act: Interactive Embodied Reasoning in Vision-Language-Action Models},
  author={Ling, Yiran and Lian, Qing and Li, Jinghang and Jiang, Qing and Zhang, Tianming and Jiang, Xiaoke and Liu, Chuanxiu and Liu, Jie and Zhang, Lei},
  journal={arXiv preprint arXiv:2605.13632},
  year={2026}
}
```

## License

This repository currently carries the Apache 2.0 license in [LICENSE](LICENSE).

## Acknowledgements

Acknowledgements will be finalized with the release, including upstream model code, datasets, simulators, and evaluation frameworks used by GTA-VLA.
