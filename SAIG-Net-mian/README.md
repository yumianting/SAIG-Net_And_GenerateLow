# SAIG-Net: Low-Light Image Enhancement

This repository contains the SAIGNet model, training and evaluation scripts, and pretrained checkpoints for low-light image enhancement. Commands below are run from the repository root.

## Quick start

Use a Python environment with PyTorch installed. The inference scripts also use NumPy, Pillow, OpenCV, PyYAML, `timm`, `einops`, `scikit-image`, and `tqdm`. Install versions compatible with your PyTorch and CUDA setup. The repository does not currently include a top-level `requirements.txt`.

The following command loads the included LOLv1 checkpoint and runs a small inference benchmark on the included image. Use `--gpu_id -1` for CPU or `--gpu_id 0` for the first CUDA GPU.

```bash
python benchmark_lolv1_speed.py \
  --gpu_id 0 \
  --image datasets/LOLv1/1.png \
  --opt Options/SAIGNet_lolv1.yml \
  --weights model_output/LOLv1/psnr_24.42_81500.pth \
  --width 128 --height 128 --warmup 1 --runs 5
```

This benchmark resizes the image to the requested dimensions and reports forward-pass time; it does not save an enhanced image. For a saved result and SAI2E feature maps, run:

```bash
python visual.py \
  --opt Options/SAIGNet_lolv1.yml \
  --weights model_output/LOLv1/psnr_24.42_81500.pth \
  --input datasets/LOLv1/1.png \
  --save_dir results/visual_lolv1 \
  --device cuda --save_result
```

The enhanced image is saved as `results/visual_lolv1/1_result.png`. Set `--device cpu` if CUDA is unavailable. `visual.py` also accepts an input directory and writes feature heatmaps alongside each result.

## Checkpoints and configurations

| Dataset | Configuration | Checkpoint |
| --- | --- | --- |
| LOLv1 | `Options/SAIGNet_lolv1.yml` | `model_output/LOLv1/psnr_24.42.pth` |
| LOLv2 Real | `Options/SAIGNet_lolv2_real.yml` | `model_output/LOLv2_real/psnr_23.38.pth` |
| LOLv2 Synthetic | `Options/SAIGNet_lolv2_syn.yml` | `model_output/LOLv2_synthetic/psnr_25.54.pth` |
| MineLOL | `Options/SAIGNet_melol_low_high.yml` | `model_output/MineLOL/psnr_25.94.pth` |

Always pass a matching configuration and checkpoint explicitly. Some scripts' built-in default checkpoint names include `best_`, but the files in this repository do not.

## Evaluate a dataset

`test_from_dataset.py` reads image folders from `datasets.val.dataroot_lq` and, when available, `datasets.val.dataroot_gt` in the selected YAML file. Prepare the dataset at those paths or edit the paths before running. Paired low-light and ground-truth files must share the same filename stem.

For LOLv1, the configuration expects `datasets/lolv1/eval15/low` and `datasets/lolv1/eval15/high`. The included `datasets/LOLv1/1.png` is only a sample image; the full evaluation dataset is not included.

```bash
python test_from_dataset.py \
  --opt Options/SAIGNet_lolv1.yml \
  --weights model_output/LOLv1/psnr_24.42_81500.pth \
  --dataset LOLv1 \
```

The script saves enhanced images under `results/<dataset>/<configuration>/<checkpoint>/` and reports PSNR and SSIM when ground truth is configured. LPIPS is enabled by default for paired evaluation; install `lpips` and remove `--no_lpips` to calculate it. For unpaired inference, clear `datasets.val.dataroot_gt` in the YAML and point `datasets.val.dataroot_lq` to an image folder; metrics requiring ground truth will be skipped.

## Train

Set the training and validation dataset paths in the selected YAML file, then run:

```bash
python train.py --opt Options/SAIGNet_lolv1.yml
```

Training settings, including batch size, total iterations, learning rate, and checkpoint frequency, are defined in the YAML file. Training requires the complete dataset and a suitable PyTorch environment; it has not been validated by the quick-start inference check.

## Repository layout

| Path | Purpose |
| --- | --- |
| `basicsr/models/archs/SAIGNet_arch.py` | SAIGNet architecture |
| `Options/` | Dataset and model configurations |
| `model_output/` | Included pretrained checkpoints |
| `test_from_dataset.py` | Folder inference and paired metrics |
| `visual.py` | Enhanced images and SAI2E feature maps |
| `benchmark_lolv1_speed.py` | Single-image inference timing |
| `train.py` | Training entry point |
