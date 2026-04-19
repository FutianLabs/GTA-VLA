# GTA-VLA

GTA-VLA is a public-facing repository skeleton for a Vision-Language-Action project.

This repository is being prepared from an internal development workspace. The current version is a curated first-pass export that keeps the core source code, configs, training and evaluation entrypoints, and metadata templates, while intentionally excluding datasets, checkpoints, experiment logs, visualization outputs, and local environment artifacts.

## Status

This repository is under active cleanup before open-source release.

Current scope:

- Core model code
- Dataset loading and preprocessing logic
- Training and evaluation scripts
- Configuration files
- Metadata examples

Not included yet:

- Public checkpoints
- Reproducibility benchmarks
- Complete installation validation
- Full documentation and examples

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

TODO:

- Describe the recommended Python version.
- Document the core dependency installation path.
- Split core dependencies and optional evaluation dependencies.
- Document any CUDA and PyTorch requirements.

## Data Preparation

TODO:

- Describe supported datasets.
- Document expected raw data layout.
- Document preprocessing commands.
- Explain the format of the metadata files under `data/`.

Important note:

- This repository does not ship raw datasets.
- Paths in local development scripts may still need to be replaced with public placeholders.

## Training

TODO:

- Add one minimal training command.
- Explain the required config and metadata arguments.
- Document output directory structure.

## Evaluation

TODO:

- Add one minimal evaluation command.
- Document optional external environments.
- Explain benchmark-specific dependencies.

## Checkpoints

TODO:

- Provide public checkpoint download links.
- Document expected checkpoint layout.

## Known Limitations

TODO:

- List unsupported tasks or incomplete modules.
- Mark any scripts that still assume internal infrastructure.

## Release Checklist

TODO:

- Audit absolute paths and internal URLs.
- Clean dependency declarations.
- Add usage examples.
- Add tests and CI.
- Add citation information.

## Citation

TODO: Add BibTeX entries and paper links.

## License

This repository currently carries the Apache 2.0 license in [LICENSE](/VLA-Data/scripts/lingyiran/GTA-VLA/LICENSE).

## Acknowledgements

TODO: Credit upstream projects, datasets, and evaluation frameworks.# GTA-VLA
