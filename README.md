# UltraUPConvNet

UltraUPConvNet is an ultrasound omni-task network that unifies semantic segmentation and classification through a ConvNeXt backbone with prompt-conditioned heads. The repository provides training and inference scripts for handling multiple ultrasound datasets with optional positional, task, type, and nature prompts.

## Features
- **Omni-task architecture:** Joint segmentation and binary/multi-class classification heads driven by ConvNeXt features and prompt encodings.
- **Prompt-aware training:** Optional prompts that encode anatomy, task type, view type, and lesion nature to guide both segmentation and classification branches.
- **Flexible data loaders:** Unified dataset classes for segmentation and classification, with weighted sampling and strong augmentations.
- **Reproducible configs:** YACS configuration system for model hyperparameters and Swin/ConvNeXt backbones.

## Repository Structure
- `omni_train.py` & `omni_trainer.py`: Entry point and training loop for omni-task learning with prompts, data loaders, losses, and logging.
- `omni_test.py`: Evaluation and inference pipeline with optional result saving.
- `model.py`: Standalone inference helper that loads a checkpoint and runs both segmentation and classification on a provided JSON split.
- `networks/`: ConvNeXt backbone (`convnext.py`), prompt-enabled omni-task model (`omni_convnext_transformer.py`), and decoder head (`decoder_UPerhead.py`).
- `datasets/`: Data utilities, augmentations, and omni-task dataset definitions for segmentation/classification splits.
- `configs/`: Swin/ConvNeXt configuration presets (e.g., `swin_tiny_patch4_window7_224_lite.yaml`).
- `baseline.sh`: Example shell commands for training, resuming, and testing.

## Installation
1. Create and activate a Python environment (Python ≥3.8 is recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation
The code expects a root directory (default `data/`) containing separate folders for segmentation and classification tasks.

```
<data_root>
├── segmentation/
│   ├── <dataset_name>/
│   │   ├── imgs/                # Images listed in train/val/test .txt files
│   │   ├── masks/               # Corresponding masks (values mapped via config.yaml)
│   │   ├── train.txt | val.txt | test.txt
│   │   └── config.yaml          # Label mapping used to remap mask values
└── classification/
    ├── <dataset_name>/
    │   ├── <class_id>/image.png # Images organized by class subfolders
    │   ├── train.txt | val.txt | test.txt (paths relative to classification/)
    │   └── (optional) masks from segmentation/ for prompt augmentation
```

Supported dataset keys include public and private ultrasound sets such as `BUS-BRA`, `BUSIS`, `CAMUS`, `DDTI`, `Fetal_HC`, `KidneyUS`, `Appendix`, `Fatty-Liver`, and corresponding private splits. When prompts are enabled, the loaders attach one-hot encodings for task, anatomy, nature, and patch type to each sample.

## Training
Use `omni_train.py` to launch training. Key arguments include `--root_path`, `--output_dir`, `--batch_size`, `--base_lr`, `--prompt`, and `--use_pretrained_model`.

```bash
export CUDA_VISIBLE_DEVICES=0
python omni_train.py \
  --root_path=data/ \
  --output_dir=exp_out/trial_1 \
  --batch_size=16 \
  --base_lr=0.003 \
  --prompt \
  --use_pretrained_model \
  --pretrained_path=/path/to/convnext_base_22k_1k_224.pth
```

- Resume training with `--resume /path/to/checkpoint.pth`.
- Adjust learning rate automatically scales with `--batch_size` multiples of 6.
- Set `--prompt --adapter_ft` to freeze non-prompt layers during fine-tuning.

## Evaluation / Inference
Run `omni_test.py` to evaluate a checkpoint. Use `--is_saveout` to store predictions.

```bash
python omni_test.py \
  --root_path=data/ \
  --output_dir=exp_out/trial_1 \
  --batch_size=16 \
  --prompt \
  --use_pretrained_model \
  --resume=exp_out/trial_1/best_model.pth \
  --is_saveout
```

For JSON-driven inference across mixed tasks, use `model.py` with a prepared `data_list` describing each case and an input directory mirroring that JSON:

```bash
python model.py
```

The script loads `exp_out/trial_2/best_model.pth` by default and writes segmentation masks plus `classification.json` into `exp_out/sample_result_submission`.

## Configuration
Model and data hyperparameters are managed via YACS configs. The default Swin setup lives in `configs/swin_tiny_patch4_window7_224_lite.yaml`, while ConvNeXt options (drop path, prompt toggles, etc.) are set through CLI flags and code defaults in `config.py` and `omni_train.py`.

## Tips
- Ensure `config.py` points `MODEL.PRETRAIN_CKPT` to available weights or pass `--pretrained_path` for ConvNeXt.
- Mixed precision is available via `--amp-opt-level` (O0/O1/O2).
- Set `CUDA_VISIBLE_DEVICES` to control GPUs; deterministic training is toggled via `--deterministic`.

## Citation
If you use UltraUPConvNet in your research, please cite the corresponding paper once available.
