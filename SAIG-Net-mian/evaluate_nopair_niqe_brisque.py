import argparse
import csv
import os
import re
from copy import deepcopy
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from basicsr.metrics import calculate_niqe
from basicsr.models.archs import define_network
from basicsr.utils.options import parse

try:
    from brisque import BRISQUE
except ImportError:
    BRISQUE = None


IMG_EXTS = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Enhance Nopair images and calculate NIQE/BRISQUE.'
    )
    parser.add_argument('--opt', default='Options/SAIGNet_melol_low_high.yml',
                        help='Path to SAIGNet option YAML.')
    parser.add_argument('--weights', default='./model_output/MineLOL/psnr_25.94_56000.pth',
                        help='Path to SAIGNet checkpoint.')
    parser.add_argument('--input_root', default='datasets',
                        help='Root directory of no-reference Nopair datasets.')
    parser.add_argument('--result_root', default='result/Nopair',
                        help='Directory to save enhanced images and metric CSV files.')
    parser.add_argument('--csv_name', default='metrics.csv',
                        help='Per-image metric CSV filename under result_root.')
    parser.add_argument('--summary_name', default='summary.csv',
                        help='Per-dataset summary CSV filename under result_root.')
    parser.add_argument('--gpu_id', default='0',
                        help='GPU id. Use -1 for CPU.')
    parser.add_argument('--param_key', default='params',
                        help='Checkpoint state-dict key, for example params or params_ema.')
    parser.add_argument('--non_strict_load', action='store_true',
                        help='Load checkpoint with strict=False.')
    parser.add_argument('--factor', type=int, default=8,
                        help='Pad image height/width to multiples of this factor. SAIGNet needs 8 for three downsample stages.')
    parser.add_argument('--self_ensemble', action='store_true',
                        help='Use x8 self-ensemble during inference.')
    parser.add_argument('--no_highlight_protection', action='store_true',
                        help='Disable highlight protection after enhancement (enabled by default).')
    parser.add_argument('--highlight_strength', type=float, default=0.65,
                        help='Maximum original-image blend weight in [0, 1]; 0 disables protection.')
    parser.add_argument('--highlight_low', type=float, default=0.35,
                        help='Input luminance at which highlight protection starts, in [0, 1].')
    parser.add_argument('--highlight_high', type=float, default=0.70,
                        help='Input luminance at which highlight protection reaches full strength.')
    parser.add_argument('--highlight_radius', type=int, default=4,
                        help='Radius of edge-aware mask smoothing; 0 skips smoothing. '
                             'Highlight options are not applied with --metrics_only.')
    parser.add_argument('--no_color_preservation', action='store_true',
                        help='Disable input RGB-ratio preservation after enhancement '
                             '(enabled by default; independent of highlight protection).')
    parser.add_argument('--exposure_strength', type=float, default=0.45,
                        help='Brightness change retained by color preservation, in [0, 1]. '
                             '0 returns the original; 1 requests full enhanced luminance '
                             'subject to RGB gamut limits. Ignored with --metrics_only '
                             'or --no_color_preservation.')
    parser.add_argument('--crop_border', type=int, default=0,
                        help='Crop border used by NIQE.')
    parser.add_argument('--niqe_convert_to', default='y', choices=['y', 'gray'],
                        help='Color conversion used by NIQE.')
    parser.add_argument('--brisque_input', default='path', choices=['path', 'bgr', 'rgb'],
                        help='Input format passed to BRISQUE.score(). Use path for saved result images.')
    parser.add_argument('--image_ext', default='.png',
                        help='Extension used when saving enhanced images.')
    parser.add_argument('--metrics_only', action='store_true',
                        help='Skip enhancement and calculate metrics from existing result_root images.')
    return parser.parse_args()


def natural_key(path):
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', name)]


def resolve_path(path):
    path = os.path.expanduser(path)
    if os.path.exists(path):
        return path
    candidates = []
    if 'NOpair' in path:
        candidates.append(path.replace('NOpair', 'Nopair'))
    if 'Nopair' in path:
        candidates.append(path.replace('Nopair', 'NOpair'))
    if 'lolv1' in path:
        candidates.append(path.replace('lolv1', 'LOLv1'))
    if 'LOLv1' in path:
        candidates.append(path.replace('LOLv1', 'lolv1'))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def is_valid_image_path(path):
    name = os.path.basename(path)
    parts = path.split(os.sep)
    return (
        os.path.splitext(name)[1].lower() in IMG_EXTS
        and not name.startswith('._')
        and '__MACOSX' not in parts
    )


def list_images(root):
    root = resolve_path(root)
    if not os.path.isdir(root):
        raise FileNotFoundError('Image root not found: %s' % root)
    paths = [path for path in glob(os.path.join(root, '**', '*'), recursive=True)
             if os.path.isfile(path) and is_valid_image_path(path)]
    paths.sort(key=lambda x: (dataset_name(root, x), natural_key(x)))
    if not paths:
        raise RuntimeError('No images found in: %s' % root)
    return paths


def dataset_name(root, path):
    rel = os.path.relpath(path, root)
    return rel.split(os.sep)[0]


def output_path_for(input_root, input_path, result_root, image_ext):
    rel_parts = os.path.relpath(input_path, input_root).split(os.sep)
    if len(rel_parts) >= 2 and rel_parts[1] == rel_parts[0]:
        rel_parts.pop(1)
    rel_parts[-1] = os.path.splitext(rel_parts[-1])[0] + image_ext
    return os.path.join(result_root, *rel_parts)


def extract_state_dict(checkpoint, param_key):
    if isinstance(checkpoint, dict):
        if param_key and param_key in checkpoint:
            checkpoint = checkpoint[param_key]
        else:
            for key in ('params', 'params_ema', 'state_dict', 'model', 'net'):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
    if not isinstance(checkpoint, dict):
        raise RuntimeError('Checkpoint does not contain a state dict.')
    return {key[7:] if key.startswith('module.') else key: value
            for key, value in checkpoint.items()}


def build_model(opt_path, weights_path, args, device):
    opt = parse(opt_path, is_train=False)
    model = define_network(deepcopy(opt['network_g'])).to(device)
    weights_path = resolve_path(weights_path)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError('Checkpoint not found: %s' % weights_path)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = extract_state_dict(checkpoint, args.param_key)
    incompatible = model.load_state_dict(state_dict, strict=not args.non_strict_load)
    if args.non_strict_load:
        print('Missing keys: %d' % len(incompatible.missing_keys))
        print('Unexpected keys: %d' % len(incompatible.unexpected_keys))
    model.eval()
    return model


def load_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('Failed to read image: %s' % path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def rgb_to_tensor(image, device):
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0).to(device)


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def guided_smooth_mask(guide, mask, radius, eps=1e-3):
    """Smooth a mask with a grayscale guided filter; arrays are float32 in [0, 1]."""
    if radius == 0:
        return mask
    kernel = (2 * radius + 1, 2 * radius + 1)

    def mean(image):
        return cv2.boxFilter(image, -1, kernel, borderType=cv2.BORDER_REFLECT)

    mean_guide = mean(guide)
    mean_mask = mean(mask)
    variance = np.maximum(mean(guide * guide) - mean_guide * mean_guide, 0)
    covariance = mean(guide * mask) - mean_guide * mean_mask
    slope = covariance / (variance + eps)
    intercept = mean_mask - slope * mean_guide
    return np.clip(mean(slope) * guide + mean(intercept), 0, 1)


def protect_highlights(original, enhanced, strength=0.8, low=0.55,
                       high=0.85, radius=4):
    """Blend brightened highlights back toward the original RGB image.

    Inputs are equal-sized HWC float32 RGB images in [0, 1]. The returned
    mask is the original-image blend weight. This cannot recover highlights
    already clipped in the original. Thresholds are in display RGB space.
    """
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Highlight strength must be finite and in [0, 1].')
    if not np.isfinite([low, high]).all() or not 0 <= low < high <= 1:
        raise ValueError('Highlight thresholds must satisfy 0 <= low < high <= 1.')
    if not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError('Highlight radius must be a nonnegative integer.')
    if original.shape != enhanced.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError('Highlight protection requires matching HWC RGB images.')
    if strength == 0:
        return enhanced, np.zeros(enhanced.shape[:2], dtype=np.float32)

    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    input_luma = original @ weights
    output_luma = enhanced @ weights
    # The input gate protects shadows. The gain gate avoids undoing a model's
    # correct darkening and smoothly reaches full weight at a +0.15 luma gain.
    input_gate = smoothstep((input_luma - low) / (high - low))
    gain_gate = smoothstep((output_luma - input_luma) / 0.15)
    mask = input_gate * gain_gate
    mask = guided_smooth_mask(input_luma, mask, radius)
    # Reapply pointwise gates: smoothing must not spill into shadows or pixels
    # that were not brightened, even across a strong brightness boundary.
    mask = strength * np.minimum(mask, np.minimum(input_gate, gain_gate))
    output = enhanced + mask[..., None] * (original - enhanced)
    return np.clip(output, 0, 1), mask


def preserve_input_color(original, enhanced, strength=0.45):
    """Transfer a controlled luminance change using one gain for all RGB channels.

    HWC RGB float32 inputs must have equal shapes and be in [0, 1]. Keeping
    source RGB ratios avoids mixing incompatible source/model chromaticities.
    A shared gain limit leaves highlight headroom without clipping channels
    separately. Source color casts, noise, and completely black pixels remain.
    """
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Exposure strength must be finite and in [0, 1].')
    if original.shape != enhanced.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError('Color preservation requires matching HWC RGB images.')
    if strength == 0:
        return original.copy()
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    input_luma = original @ weights
    target_luma = input_luma + strength * (enhanced @ weights - input_luma)
    gain = target_luma / np.maximum(input_luma, 1e-6)
    peak = original.max(axis=2)
    peak_limit = peak + strength * (1 - peak)
    gain = np.minimum(gain, peak_limit / np.maximum(peak, 1e-6))
    return np.clip(original * gain[..., None], 0, 1)


def self_ensemble(x, model):
    def forward_transformed(inp, hflip, vflip, rotate):
        if hflip:
            inp = torch.flip(inp, (-2,))
        if vflip:
            inp = torch.flip(inp, (-1,))
        if rotate:
            inp = torch.rot90(inp, dims=(-2, -1))
        out = model(inp)
        if rotate:
            out = torch.rot90(out, dims=(-2, -1), k=3)
        if vflip:
            out = torch.flip(out, (-1,))
        if hflip:
            out = torch.flip(out, (-2,))
        return out

    outputs = []
    for hflip in (False, True):
        for vflip in (False, True):
            for rotate in (False, True):
                outputs.append(forward_transformed(x, hflip, vflip, rotate))
    return torch.stack(outputs).mean(dim=0)


def infer_image(model, input_tensor, factor, use_self_ensemble, highlight_options=None,
                color_options=None):
    original_tensor = input_tensor
    _, _, h, w = input_tensor.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h or pad_w:
        input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    if h < 3000 and w < 3000:
        output = self_ensemble(input_tensor, model) if use_self_ensemble else model(input_tensor)
    else:
        input_odd = input_tensor[:, :, :, 1::2]
        input_even = input_tensor[:, :, :, 0::2]
        output_odd = self_ensemble(input_odd, model) if use_self_ensemble else model(input_odd)
        output_even = self_ensemble(input_even, model) if use_self_ensemble else model(input_even)
        output = torch.zeros_like(input_tensor)
        output[:, :, :, 1::2] = output_odd
        output[:, :, :, 0::2] = output_even

    output = torch.clamp(output[:, :, :h, :w], 0, 1)
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if highlight_options is not None or color_options is not None:
        original = original_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if highlight_options is not None and highlight_options.get('strength', 0.8) > 0:
        output, _ = protect_highlights(original, output, **highlight_options)
    if color_options is not None:
        output = preserve_input_color(original, output, **color_options)
    return np.round(output * 255.0).astype(np.uint8)


def build_brisque_metric():
    if BRISQUE is None:
        raise ImportError(
            'The `brisque` package is required for BRISQUE calculation. '
            'Install it in your environment, for example: pip install brisque'
        )

    try:
        return BRISQUE(url=False)
    except TypeError:
        return BRISQUE()


def calculate_brisque(metric, rgb_u8, image_path, args):
    if args.brisque_input == 'path':
        score_input = image_path
    elif args.brisque_input == 'bgr':
        score_input = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    else:
        score_input = rgb_u8

    try:
        score = metric.score(score_input)
    except Exception as exc:
        if args.brisque_input == 'path':
            bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
            score = metric.score(bgr)
        else:
            raise RuntimeError('BRISQUE.score() failed for %s' % image_path) from exc
    return float(np.asarray(score).squeeze())


def calculate_metrics(rgb_u8, output_path, brisque_metric, args):
    bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    niqe = float(np.asarray(calculate_niqe(
        bgr_u8,
        crop_border=args.crop_border,
        input_order='HWC',
        convert_to=args.niqe_convert_to,
    )).squeeze())
    brisque = calculate_brisque(brisque_metric, rgb_u8, output_path, args)
    return niqe, brisque


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['dataset'], []).append(row)
    summary = []
    for name in sorted(grouped):
        items = grouped[name]
        niqe_values = [float(item['niqe']) for item in items]
        brisque_values = [float(item['brisque']) for item in items]
        summary.append({
            'dataset': name,
            'count': len(items),
            'avg_niqe': '%.6f' % float(np.mean(niqe_values)),
            'avg_brisque': '%.6f' % float(np.mean(brisque_values)),
        })
    all_niqe = [float(item['niqe']) for item in rows]
    all_brisque = [float(item['brisque']) for item in rows]
    summary.append({
        'dataset': 'ALL',
        'count': len(rows),
        'avg_niqe': '%.6f' % float(np.mean(all_niqe)),
        'avg_brisque': '%.6f' % float(np.mean(all_brisque)),
    })
    return summary


def main():
    args = parse_args()
    if args.factor <= 0:
        raise RuntimeError('--factor must be greater than 0.')
    if not np.isfinite(args.highlight_strength) or not 0 <= args.highlight_strength <= 1:
        raise RuntimeError('--highlight_strength must be finite and in [0, 1].')
    if (not np.isfinite([args.highlight_low, args.highlight_high]).all()
            or not 0 <= args.highlight_low < args.highlight_high <= 1):
        raise RuntimeError('Highlight thresholds must satisfy 0 <= low < high <= 1.')
    if args.highlight_radius < 0:
        raise RuntimeError('--highlight_radius must be nonnegative.')
    if not np.isfinite(args.exposure_strength) or not 0 <= args.exposure_strength <= 1:
        raise RuntimeError('--exposure_strength must be finite and in [0, 1].')
    color_options = (None if args.no_color_preservation
                     else dict(strength=args.exposure_strength))
    highlight_options = None
    if not args.no_highlight_protection and args.highlight_strength > 0:
        highlight_options = dict(strength=args.highlight_strength, low=args.highlight_low,
                                 high=args.highlight_high, radius=args.highlight_radius)
    if not args.image_ext.startswith('.'):
        args.image_ext = '.' + args.image_ext

    input_root = resolve_path(args.input_root)
    result_root = resolve_path(args.result_root)
    if args.gpu_id != '-1' and torch.cuda.is_available():
        device = torch.device('cuda:%s' % args.gpu_id)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    brisque_metric = build_brisque_metric()
    model = None if args.metrics_only else build_model(args.opt, args.weights, args, device)

    input_paths = list_images(result_root if args.metrics_only else input_root)
    iterator = tqdm(input_paths, total=len(input_paths)) if tqdm is not None else input_paths
    rows = []

    print('Device: %s' % device)
    if not args.metrics_only:
        print('Model weights: %s' % resolve_path(args.weights))
        print('Highlight protection: %s' % (highlight_options or 'disabled'))
        print('Input color preservation: %s' % (color_options or 'disabled'))
    else:
        print('Metrics only: highlight protection and color preservation are not applied.')
    print('BRISQUE: brisque.BRISQUE')
    print('Images: %d' % len(input_paths))
    print('Result root: %s' % result_root)

    with torch.inference_mode():
        for input_path in iterator:
            if args.metrics_only:
                output_path = input_path
                rgb_u8 = load_rgb(output_path)
                current_dataset = dataset_name(result_root, output_path)
                rel_input = os.path.relpath(output_path, result_root)
            else:
                input_tensor = rgb_to_tensor(load_rgb(input_path), device)
                rgb_u8 = infer_image(model, input_tensor, args.factor, args.self_ensemble,
                                     highlight_options=highlight_options,
                                     color_options=color_options)
                output_path = output_path_for(input_root, input_path, result_root, args.image_ext)
                save_rgb(output_path, rgb_u8)
                current_dataset = dataset_name(input_root, input_path)
                rel_input = os.path.relpath(input_path, input_root)

            niqe, brisque = calculate_metrics(rgb_u8, output_path, brisque_metric, args)
            rows.append({
                'dataset': current_dataset,
                'image': rel_input,
                'output': os.path.relpath(output_path, result_root),
                'niqe': '%.6f' % niqe,
                'brisque': '%.6f' % brisque,
            })
            print('%s NIQE: %.6f BRISQUE: %.6f' % (rel_input, niqe, brisque))

    summary_rows = summarize(rows)
    write_csv(os.path.join(result_root, args.csv_name), rows,
              ['dataset', 'image', 'output', 'niqe', 'brisque'])
    write_csv(os.path.join(result_root, args.summary_name), summary_rows,
              ['dataset', 'count', 'avg_niqe', 'avg_brisque'])

    print('Dataset summary:')
    for row in summary_rows:
        print('%s count: %s NIQE: %s BRISQUE: %s' % (
            row['dataset'], row['count'], row['avg_niqe'], row['avg_brisque']))

    all_row = summary_rows[-1]
    print('Average NIQE: %s' % all_row['avg_niqe'])
    print('Average BRISQUE: %s' % all_row['avg_brisque'])
    print('Metrics CSV: %s' % os.path.join(result_root, args.csv_name))
    print('Summary CSV: %s' % os.path.join(result_root, args.summary_name))


if __name__ == '__main__':
    main()
