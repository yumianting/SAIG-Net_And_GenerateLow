import argparse
import importlib.util
import os
import re
import shutil
import sys
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

try:
    import utils
except ModuleNotFoundError as exc:
    if exc.name != 'utils':
        raise
    from Enhancement import utils

IMAGE_EXTENSIONS = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')


@dataclass
class ImageEntry:
    path: str
    rel_path: str
    split: str


def natural_key(path):
    name = os.path.basename(path)
    parts = [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', name)
    ]
    return parts + [name.lower()]


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def resolve_path(path, base_dir):
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base_dir, path))


def is_image(path):
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def is_under(path, folder):
    path = os.path.abspath(path)
    folder = os.path.abspath(folder)
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:
        return False


def collect_images_from_dir(folder, exclude_dir=''):
    if not os.path.isdir(folder):
        raise FileNotFoundError('Image directory not found: %s' % folder)
    paths = [
        path for path in glob(os.path.join(folder, '**', '*'), recursive=True)
        if os.path.isfile(path) and is_image(path)
        and not (exclude_dir and is_under(path, exclude_dir))
    ]
    return sorted(paths, key=natural_key)


def collect_images_from_txt(txt_path, base_dir):
    paths = []
    txt_dir = os.path.dirname(txt_path)
    with open(txt_path, 'r') as f:
        for line in f:
            item = line.strip()
            if not item:
                continue
            candidates = [
                resolve_path(item, base_dir),
                resolve_path(item, txt_dir),
            ]
            for candidate in candidates:
                if os.path.isfile(candidate) and is_image(candidate):
                    paths.append(candidate)
                    break
            else:
                raise FileNotFoundError(
                    'Image listed in %s was not found: %s' % (txt_path, item))
    return sorted(paths, key=natural_key)


def rel_under(path, root):
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return os.path.basename(path)
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        parts = os.path.normpath(path).split(os.sep)
        if 'images' in parts:
            return os.path.join(*parts[parts.index('images'):])
        return os.path.basename(path)
    return rel


def yolo_base_dir(data_yaml, data):
    yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
    if data.get('path'):
        return resolve_path(data['path'], yaml_dir)
    return yaml_dir


def normalize_splits(split_arg, data):
    if split_arg.lower() == 'all':
        splits = [key for key in ('train', 'val', 'test') if data.get(key)]
        if not splits:
            splits = [key for key in ('images', 'image_dir') if data.get(key)]
        return splits
    return [item.strip() for item in split_arg.split(',') if item.strip()]


def collect_yolo_entries(data_yaml, split_arg):
    data = load_yaml(data_yaml)
    base_dir = yolo_base_dir(data_yaml, data)
    yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
    entries = []

    for split in normalize_splits(split_arg, data):
        value = data.get(split)
        if value is None:
            raise KeyError('Split "%s" not found in %s' % (split, data_yaml))

        specs = value if isinstance(value, list) else [value]
        for spec in specs:
            source = resolve_path(spec, base_dir)
            if not os.path.exists(source):
                source = resolve_path(spec, yaml_dir)

            if os.path.isfile(source) and source.lower().endswith('.txt'):
                paths = collect_images_from_txt(source, base_dir)
                for path in paths:
                    rel = rel_under(path, base_dir)
                    entries.append(ImageEntry(path=path, rel_path=rel, split=split))
            elif os.path.isdir(source):
                paths = collect_images_from_dir(source)
                spec_rel = rel_under(source, base_dir)
                for path in paths:
                    rel = os.path.join(spec_rel, rel_under(path, source))
                    entries.append(ImageEntry(path=path, rel_path=rel, split=split))
            elif os.path.isfile(source) and is_image(source):
                rel = rel_under(source, base_dir)
                entries.append(ImageEntry(path=source, rel_path=rel, split=split))
            else:
                raise FileNotFoundError('YOLO split path not found: %s' % spec)

    return dedupe_entries(entries), data, base_dir


def collect_plain_entries(image_dir, exclude_dir=''):
    image_dir = os.path.abspath(image_dir)
    paths = collect_images_from_dir(image_dir, exclude_dir=exclude_dir)
    return [
        ImageEntry(path=path, rel_path=rel_under(path, image_dir), split='images')
        for path in paths
    ]


def dedupe_entries(entries):
    seen = set()
    result = []
    for entry in entries:
        real = os.path.realpath(entry.path)
        if real in seen:
            continue
        seen.add(real)
        result.append(entry)
    return result


def output_image_path(output_dir, rel_path, save_ext):
    rel_root, rel_ext = os.path.splitext(rel_path)
    ext = save_ext or rel_ext
    if not ext.startswith('.'):
        ext = '.' + ext
    return os.path.join(output_dir, rel_root + ext.lower())


def label_path_for_image(image_path):
    parts = os.path.normpath(image_path).split(os.sep)
    if 'images' not in parts:
        return None
    idx = parts.index('images')
    label_parts = list(parts)
    label_parts[idx] = 'labels'
    label_path = os.sep.join(label_parts)
    return os.path.splitext(label_path)[0] + '.txt'


def output_label_path(output_path):
    parts = os.path.normpath(output_path).split(os.sep)
    if 'images' in parts:
        idx = parts.index('images')
        parts[idx] = 'labels'
        label_path = os.sep.join(parts)
        return os.path.splitext(label_path)[0] + '.txt'
    output_dir = os.path.dirname(os.path.dirname(output_path))
    return os.path.join(output_dir, 'labels',
                        os.path.splitext(os.path.basename(output_path))[0] + '.txt')


def maybe_copy_label(input_image_path, output_image_path_):
    label_path = label_path_for_image(input_image_path)
    if label_path is None or not os.path.isfile(label_path):
        return False
    dst = output_label_path(output_image_path_)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(label_path, dst)
    return True


def find_weight_path(opt, opt_path):
    path_opt = opt.get('path', {})
    weight_path = path_opt.get('pretrain_network_g') if isinstance(path_opt, dict) else None
    if not isinstance(weight_path, str) or not weight_path:
        return ''

    weight_path = os.path.expanduser(weight_path)
    if os.path.isabs(weight_path) or os.path.isfile(weight_path):
        return weight_path

    yaml_dir_path = os.path.join(os.path.dirname(os.path.abspath(opt_path)), weight_path)
    if os.path.isfile(yaml_dir_path):
        return yaml_dir_path
    return weight_path


def extract_state_dict(checkpoint, param_key):
    if isinstance(checkpoint, dict):
        if param_key and param_key in checkpoint and isinstance(checkpoint[param_key], dict):
            checkpoint = checkpoint[param_key]
        else:
            for key in ('params', 'params_ema', 'state_dict', 'model', 'net'):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

    if not isinstance(checkpoint, dict):
        raise RuntimeError('Checkpoint does not contain a state dict.')

    state_dict = {}
    for key, value in checkpoint.items():
        clean_key = key
        if clean_key.startswith('module.'):
            clean_key = clean_key[7:]
        if clean_key.startswith('net_g.'):
            clean_key = clean_key[6:]
        state_dict[clean_key] = value
    return state_dict


def load_enhancement_model(opt_path, param_key, strict, device, weights=''):
    opt = load_yaml(opt_path)
    network_opt = deepcopy(opt['network_g'])
    network_type = network_opt.pop('type')
    if network_type != 'SAIGNet':
        raise RuntimeError('Expected SAIGNet, got %s' % network_type)
    arch_dir = Path(__file__).resolve().parent / 'basicsr' / 'models' / 'archs'
    spec = importlib.util.spec_from_file_location(
        '_saignet_enhance_archs', arch_dir / 'SAIGNet_arch.py',
        submodule_search_locations=[str(arch_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.SAIGNet(**network_opt).to(device)

    weight_path = weights or find_weight_path(opt, opt_path)
    if not weight_path:
        raise RuntimeError(
            'No weight file configured. Set path.pretrain_network_g in the model YAML.')
    if not os.path.isfile(weight_path):
        raise FileNotFoundError('Weight file not found: %s' % weight_path)

    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = extract_state_dict(checkpoint, param_key)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model, opt, weight_path


def load_rgb_float(path):
    image = utils.load_img(path)
    return np.float32(image) / 255.0


def to_uint8(image):
    return np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)


def enhance_image(model, image, device, factor):
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, h, w = tensor.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), 'reflect')

    with torch.inference_mode():
        restored = model(tensor)

    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0, 1)
    return restored.squeeze(0).permute(1, 2, 0).cpu().numpy()


def common_output_spec(paths):
    if not paths:
        return ''
    common = os.path.commonpath(paths)
    if os.path.splitext(common)[1].lower() in IMAGE_EXTENSIONS:
        common = os.path.dirname(common)
    return common or '.'


def write_yolo_yaml(output_dir, source_data, split_outputs):
    data = deepcopy(source_data)
    data['path'] = os.path.abspath(output_dir)
    for split in ('train', 'val', 'test'):
        if split in split_outputs:
            data[split] = common_output_spec(split_outputs[split])
        elif split in data:
            data.pop(split)
    out_yaml = os.path.join(output_dir, 'enhanced_data.yaml')
    os.makedirs(output_dir, exist_ok=True)
    with open(out_yaml, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out_yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Enhance all images under datasets with the weight configured in SAIGNet YAML.'
    )
    parser.add_argument('--data-yaml', default='',
                        help='YOLO data.yaml. Supports train/val/test image dirs or txt manifests.')
    parser.add_argument('--images-dir', default='datasets',
                        help='Image directory to enhance recursively. Default: datasets.')
    parser.add_argument('--split', default='all',
                        help='YOLO split to process: train, val, test, comma-list, or all.')
    parser.add_argument('--opt', default='Options/SAIGNet_melol_low_high.yml',
                        help='SAIGNet option YAML. The checkpoint is read from path.pretrain_network_g.')
    parser.add_argument('--weights', default='',
                        help='Checkpoint path. Overrides path.pretrain_network_g in the model YAML.')
    parser.add_argument('--param-key', default='params',
                        help='Checkpoint key, for example params or params_ema.')
    parser.add_argument('--non-strict-load', action='store_true',
                        help='Allow missing/unexpected checkpoint keys.')
    parser.add_argument('--output-dir', default='results/datasets_enhanced',
                        help='Directory for enhanced images.')
    parser.add_argument('--save-ext', default='.png',
                        help='Output image extension. Use empty string to keep original extensions.')
    parser.add_argument('--device', default='auto',
                        help='auto, cpu, cuda, or cuda:0.')
    parser.add_argument('--factor', type=int, default=4,
                        help='Pad image sides to multiples of this value before inference.')
    parser.add_argument('--copy-labels', action='store_true',
                        help='Copy matching YOLO labels from labels/... to output labels/....')
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def main():
    args = parse_args()
    if args.factor <= 0:
        raise RuntimeError('--factor must be greater than 0.')

    device = choose_device(args.device)
    model, _, weight_path = load_enhancement_model(
        args.opt, args.param_key,
        strict=not args.non_strict_load, device=device, weights=args.weights)

    if args.data_yaml:
        entries, yolo_data, _ = collect_yolo_entries(args.data_yaml, args.split)
    else:
        entries = collect_plain_entries(args.images_dir, exclude_dir=args.output_dir)
        yolo_data = None

    if not entries:
        raise RuntimeError('No images found to enhance.')

    os.makedirs(args.output_dir, exist_ok=True)
    copied_labels = 0
    split_outputs = {}

    print('Model YAML: %s' % args.opt)
    print('Weights: %s' % weight_path)
    print('Device: %s' % device)
    print('Images: %d' % len(entries))
    print('Output: %s' % os.path.abspath(args.output_dir))

    for entry in tqdm(entries, total=len(entries)):
        image = load_rgb_float(entry.path)
        restored = enhance_image(model, image, device, args.factor)
        out_path = output_image_path(args.output_dir, entry.rel_path, args.save_ext)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        utils.save_img(out_path, to_uint8(restored))
        split_outputs.setdefault(entry.split, []).append(
            os.path.relpath(out_path, args.output_dir))

        if args.copy_labels and maybe_copy_label(entry.path, out_path):
            copied_labels += 1

    if yolo_data is not None:
        enhanced_yaml = write_yolo_yaml(args.output_dir, yolo_data, split_outputs)
        print('Enhanced YOLO YAML: %s' % enhanced_yaml)

    if args.copy_labels:
        print('Copied labels: %d' % copied_labels)
    print('Done.')


if __name__ == '__main__':
    main()
