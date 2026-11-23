# Low-Light Image Synthesis and Enhancement with SAIG-Net

This project combines synthetic low-light image generation with SAIG-Net enhancement. Generate low-light images from normally exposed photographs, enhance existing low-light images with a pretrained checkpoint, or use generated image pairs to train and evaluate the model.

| Directory | Purpose |
| --- | --- |
| [`GenerateLow/`](GenerateLow/) | Synthesizes low-light PNG and JPEG images and records degradation settings, seeds, and crop coordinates in a JSON manifest. |
| [`SAIG-Net-mian/`](SAIG-Net-mian/) | Contains the SAIG-Net model, pretrained checkpoints, and scripts for enhancement, training, evaluation, and feature visualization. |

## Setup

Use Python 3.10 or later in a virtual environment for the full workflow. Image synthesis requires NumPy and OpenCV and can also run on Python 3.9 with compatible package versions.

```sh
python -m pip install numpy opencv-python
```

For SAIG-Net, install matching PyTorch (`torch`) and TorchVision (`torchvision`) builds for your CPU or CUDA environment, then install the supporting packages.

```sh
python -m pip install einops lmdb pillow pyyaml scikit-image scipy timm tqdm
```

Install `lpips` if you want LPIPS for paired evaluation, and `brisque` if you want to use the no-reference evaluation script.

## Quick start

Start in the directory containing this README. Generate a low-light version of the included sample image.

```sh
python GenerateLow/generate_lowlight.py --source GenerateLow/Input/1.png --output GenerateLow/Output/demo_low.png
```

The generator writes `demo_low.png` and `demo_low_generation.json` to `GenerateLow/Output/`. Choose a fresh output filename when rerunning the generator. By default, it center-crops both dimensions to multiples of 16. Pass `--size-multiple 1` to keep the original dimensions.

Enter the SAIG-Net directory and enhance the generated image with the included MineLOL checkpoint.

```sh
cd SAIG-Net-mian
python visual.py --opt Options/SAIGNet_melol_low_high.yml --weights model_output/MineLOL/psnr_25.94_56000.pth --input ../GenerateLow/Output/demo_low.png --save_dir results/quickstart --device cpu --save_result
```

The enhanced image is saved as `results/quickstart/demo_low_result.png` alongside the SAI2E feature heatmaps. Use `--device cuda` with a CUDA-enabled PyTorch installation. The `--input` option accepts either an image or a directory of images.

## Generate paired training data

From the top-level directory, prepare a `source_dataset/` folder with at least one normally exposed image in both `train/high/` and `test/high/`. Then run the generator with a new output directory outside `source_dataset/`.

```sh
python GenerateLow/generate_lowlight.py --source source_dataset --output SAIG-Net-mian/datasets/generated_pairs --seed 20260430
```

The output directory contains `train/low/`, `train/high/`, `test/low/`, `test/high/`, and `generation_manifest.json`. Each low-light image has the same filename and crop as its normally exposed counterpart. Low-light synthesis combines gamma darkening, spatially varying illumination, vignetting, color shifts, optional blur, and simulated sensor noise. For the available controls, see the [generation guide](GenerateLow/README.md).

## Train and evaluate

Run the following commands from `SAIG-Net-mian/`. To train on the generated pairs, copy `Options/SAIGNet_melol_low_high.yml` to `Options/SAIGNet_generated.yml`. Set the following fields in the copy and retain the other settings. The configuration uses 256-pixel training crops, so make sure each training image is at least 256 pixels in both dimensions after generation.

| YAML field | Value |
| --- | --- |
| `name` | `SAIGNet_generated` |
| `datasets.train.dataroot_lq` | `./datasets/generated_pairs/train/low` |
| `datasets.train.dataroot_gt` | `./datasets/generated_pairs/train/high` |
| `datasets.val.dataroot_lq` | `./datasets/generated_pairs/test/low` |
| `datasets.val.dataroot_gt` | `./datasets/generated_pairs/test/high` |
| `path.root` | `./model_output` |

Start training with the copied configuration in a CUDA-enabled environment. The default perceptual loss uses pretrained VGG19 weights, which TorchVision can download when they are unavailable locally.

```sh
python train.py --opt Options/SAIGNet_generated.yml
```

Training writes checkpoints to `model_output/experiments/SAIGNet_generated/models/`.

To evaluate an included checkpoint on a benchmark, place the full dataset at the paths specified in the matching YAML or update those paths. For paired evaluation, the low-light inputs and ground-truth images must share the same filename stems. For example, the LOLv1 configuration expects `datasets/lolv1/eval15/low/` and `datasets/lolv1/eval15/high/`.

```sh
python test_from_dataset.py --opt Options/SAIGNet_lolv1.yml --weights model_output/LOLv1/psnr_24.42.pth --dataset LOLv1 --no_lpips
```

The script reports PSNR and SSIM and saves enhanced images under `results/<dataset>/<configuration>/<checkpoint>/`. Omit `--no_lpips` after installing `lpips` to also calculate LPIPS. To evaluate a checkpoint from your own training run, pass that checkpoint and the YAML used to train it.

## Included configurations and checkpoints

Paths in this table are relative to `SAIG-Net-mian/`.

| Dataset | Configuration | Checkpoint |
| --- | --- | --- |
| LOLv1 | `Options/SAIGNet_lolv1.yml` | `model_output/LOLv1/psnr_24.42.pth` |
| LOLv2 Real | `Options/SAIGNet_lolv2_real.yml` | `model_output/LOLv2_real/psnr_23.38.pth` |
| LOLv2 Synthetic | `Options/SAIGNet_lolv2_syn.yml` | `model_output/LOLv2_synthetic/psnr_25.54.pth` |
| MineLOL | `Options/SAIGNet_melol_low_high.yml` | `model_output/MineLOL/psnr_25.94.pth` |

## Additional tools

| Script | Purpose |
| --- | --- |
| `benchmark_lolv1_speed.py` | Measures model forward-pass time at a specified image size. |
| `evaluate_nopair_niqe_brisque.py` | Computes NIQE and BRISQUE for unpaired images and writes per-image and summary CSVs. |
| `enhance_yolo_dataset.py` | Enhances YOLO dataset images and can copy the corresponding labels. |

See the [SAIG-Net guide](SAIG-Net-mian/README.md) for an example speed benchmark and more details on model usage.
