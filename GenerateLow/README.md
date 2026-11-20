# Low-Light Image Generation

Use `generate_lowlight.py` to create a low-light image from one normally exposed image or a dataset of `high` images. The script accepts PNG, JPG, and JPEG files. It saves the generated image and a JSON manifest containing the configuration used.

## Requirements

Python 3.9 or newer, NumPy, and OpenCV are required. Install the packages if needed:

```bash
python3 -m pip install numpy opencv-python
```

Run the following commands from this project directory.

## Generate one image

The sample input is `Input/1.png`:

```bash
python3 generate_lowlight.py --source Input/1.png --output Output/1_low_v2.png
```

This creates `Output/1_low_v2.png` and `Output/1_low_v2_generation.json`. An earlier result, `Output/1_low.png`, is already present. The script refuses to overwrite an existing output image or manifest, so choose a new output name for each run.

By default, the script center-crops the width and height to multiples of 16. To preserve the original dimensions, set `--size-multiple 1`:

```bash
python3 generate_lowlight.py --source Input/1.png --output Output/1_low_full.png --size-multiple 1
```

## Generate a dataset

The input directory must contain at least one image in each split:

```text
dataset_root/
├── train/high/*.png
└── test/high/*.png
```

Run:

```bash
python3 generate_lowlight.py --source dataset_root --output new_output_root
```

The output contains `train/high`, `train/low`, `test/high`, `test/low`, and `generation_manifest.json`. Images that do not need cropping are copied into `high`; otherwise, `high` and `low` use the same center crop. The output directory must not already exist or be nested within the input directory.

## Command-line options

| Option | Default | Description |
| --- | ---: | --- |
| `--seed` | `20260430` | Random seed. Identical inputs, settings, and runtime environments produce reproducible results. |
| `--seed-mode` | `stream` | `stream` uses one continuous random stream across a dataset. `per-image` seeds image *i* with `seed + i`, starting at zero. |
| `--size-multiple` | `16` | Center-crop each dimension to a multiple of this value. Use `1` to preserve dimensions. |
| `--brightness-scale` | `1.0` | Multiply both endpoints of the brightness-factor range by this value. |
| `--noise-scale` | `1.0` | Multiply both Gaussian noise standard deviation ranges by this value and divide the Poisson levels range by its square. Smaller values generally produce less injected noise. |
| `--blur-prob` | `0.25` | Probability of applying an additional Gaussian blur, from 0 to 1. Use `0` to disable it. |

For example, create a full-size image with higher brightness and lower noise:

```bash
python3 generate_lowlight.py \
  --source Input/1.png --output Output/1_low_mild.png \
  --size-multiple 1 \
  --brightness-scale 1.08 --noise-scale 0.8
```

## Default degradation process

The script applies these steps in order: normalize pixel values to `[0, 1]`, apply gamma darkening, scale brightness, add nonuniform illumination and vignetting, apply a color profile, optionally blur, add darkness-dependent Gaussian and Poisson noise, suppress excessive red speckles, and convert to 8-bit pixels. Continuous parameters are sampled uniformly for each image. Noise values are sampled per pixel.

| Parameter | Default sampling range or probability |
| --- | --- |
| Gamma | Uniform from 1.55 to 2.15 |
| Brightness factor | Uniform from 0.38 to 0.70 |
| Base illumination | Uniform from 0.38 to 0.62 |
| Gaussian illumination spots | 1, 2, or 3 with equal probability; the first spot uses the lamp profile with 80% probability |
| Color profile | Cold, greenish, or yellow-green with equal probability; channel gains are sampled uniformly within the selected profile's ranges |
| Additional blur | 25% probability; kernel size 3 or 5 with equal probability; standard deviation uniform from 0.4 to 1.1 |
| Luminance Gaussian noise standard deviation | Uniform from 0.0025 to 0.0100 |
| Chromatic Gaussian noise standard deviation | Uniform from 0.00015 to 0.00080 |
| Poisson levels | Uniform from 36 to 68 |

Each output manifest records the full configuration, image dimensions, crop coordinates, and seed information. JPEG output uses quality 95. PNG is recommended for training data.
