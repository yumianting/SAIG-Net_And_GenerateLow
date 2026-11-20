import argparse
import importlib.util
import os
import statistics
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml


IMG_EXTS = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')
BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark SAIGNet inference time on one LOLv1 test image.'
    )
    parser.add_argument('--gpu_id', default='0', help='GPU ID, or -1 for CPU')
    parser.add_argument('--data_dir', default='./datasets/lolv1/eval15/low',
                        help='test image directory used when --image is not set')
    parser.add_argument('--image', default=None, help='single image path to benchmark')
    parser.add_argument('--opt', default='Options/SAIGNet_lolv1.yml',
                        help='model option YAML')
    parser.add_argument('--weights', default='./model_output/LOLv1/best_psnr_24.42_81500.pth',
                        help='checkpoint path. Overrides YAML path.pretrain_network_g')
    parser.add_argument('--param_key', default='params',
                        help='checkpoint key, such as params or params_ema')
    parser.add_argument('--non_strict_load', action='store_true',
                        help='allow missing or unexpected checkpoint keys')
    parser.add_argument('--width', type=int, default=600, help='benchmark image width')
    parser.add_argument('--height', type=int, default=400, help='benchmark image height')
    parser.add_argument('--warmup', type=int, default=10, help='warmup iterations')
    parser.add_argument('--runs', type=int, default=50, help='timed iterations')
    parser.add_argument('--threads', type=int, default=0,
                        help='CPU torch thread count. 0 keeps current setting')
    return parser.parse_args()


def resolve_data_dir(data_dir):
    if os.path.isdir(data_dir):
        return data_dir

    candidates = []
    if 'lolv1' in data_dir:
        candidates.append(data_dir.replace('lolv1', 'LOLv1'))
    if 'LOLv1' in data_dir:
        candidates.append(data_dir.replace('LOLv1', 'lolv1'))

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    raise RuntimeError('Image directory not found: %s' % data_dir)


def list_images(image_dir):
    image_dir = resolve_data_dir(image_dir)
    paths = [
        path for path in glob(os.path.join(image_dir, '*'))
        if os.path.splitext(path)[1].lower() in IMG_EXTS
    ]
    paths.sort()
    return paths


def pick_image(args):
    if args.image:
        if os.path.isfile(args.image):
            return args.image

        image_dir = resolve_data_dir(args.data_dir)
        image_path = os.path.join(image_dir, args.image)
        if os.path.isfile(image_path):
            return image_path

        raise RuntimeError('Image not found: %s' % args.image)

    images = list_images(args.data_dir)
    if not images:
        raise RuntimeError('No test images found in: %s' % args.data_dir)
    return images[0]


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
        raise RuntimeError('Checkpoint does not contain a state dict')

    state_dict = {}
    for key, value in checkpoint.items():
        clean_key = key[7:] if key.startswith('module.') else key
        state_dict[clean_key] = value
    return state_dict


def load_local_architecture():
    # Load the repository's architecture without importing an installed basicsr.
    arch_dir = Path(__file__).resolve().parent / 'basicsr' / 'models' / 'archs'
    spec = importlib.util.spec_from_file_location(
        '_saignet_benchmark_archs', arch_dir / 'SAIGNet_arch.py',
        submodule_search_locations=[str(arch_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SAIGNet


def load_model(args, device):
    with open(args.opt, 'r') as opt_file:
        opt = yaml.safe_load(opt_file)
    network_opt = dict(opt['network_g'])
    network_type = network_opt.pop('type')
    if network_type != 'SAIGNet':
        raise RuntimeError('Expected SAIGNet in %s, got %s' % (args.opt, network_type))
    model = load_local_architecture()(**network_opt).to(device)

    weight_path = args.weights or opt.get('path', {}).get('pretrain_network_g')
    if weight_path:
        checkpoint = torch.load(weight_path, map_location=device)
        state_dict = extract_state_dict(checkpoint, args.param_key)
        model.load_state_dict(state_dict, strict=not args.non_strict_load)
        print('Loaded checkpoint: %s' % weight_path)
    else:
        print('Warning: no checkpoint loaded. Use --weights for real inference speed.')

    model.eval()
    return model


def image_to_tensor(image_path, size, device):
    image = Image.open(image_path).convert('RGB').resize(size, BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0).to(device)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_inference(model, input_tensor):
    return model(input_tensor)


def timed_run(model, input_tensor, device):
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = run_inference(model, input_tensor)
    synchronize(device)
    return time.perf_counter() - start, output


def main():
    args = parse_args()
    if args.runs <= 0:
        raise RuntimeError('--runs must be greater than 0')
    if args.warmup < 0:
        raise RuntimeError('--warmup must be 0 or greater')
    if args.width <= 0 or args.height <= 0:
        raise RuntimeError('--width and --height must be greater than 0')
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.gpu_id != '-1' and torch.cuda.is_available():
        device = torch.device('cuda:%s' % args.gpu_id)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    image_path = pick_image(args)
    size = (args.width, args.height)
    model = load_model(args, device)
    input_tensor = image_to_tensor(image_path, size, device)

    print('Benchmark image: %s' % image_path)
    print('Benchmark size: %dx%d' % (args.width, args.height))
    print('Device: %s' % device)
    print('Warmup runs: %d' % args.warmup)
    print('Timed runs: %d' % args.runs)
    print('Timing includes preprocess: no')

    for _ in range(args.warmup):
        timed_run(model, input_tensor, device)

    times = []
    for _ in range(args.runs):
        elapsed, _ = timed_run(model, input_tensor, device)
        times.append(elapsed * 1000.0)

    avg_ms = statistics.mean(times)
    print('Average: %.3f ms' % avg_ms)
    print('Median: %.3f ms' % statistics.median(times))
    print('Min: %.3f ms' % min(times))
    print('Max: %.3f ms' % max(times))
    print('FPS: %.2f' % (1000.0 / avg_ms))


if __name__ == '__main__':
    main()
