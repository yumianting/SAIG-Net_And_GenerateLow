import argparse
import os
import re
from copy import deepcopy
from types import MethodType

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from natsort import natsorted
except ImportError:
    natsorted = None


import utils

from basicsr.models.archs import define_network
from basicsr.models.archs.illu_fea import SAI2E
from basicsr.utils.options import parse


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def natural_key(path):
    name = os.path.basename(path)
    parts = [int(part) if part.isdigit() else part.lower()
             for part in re.split(r'(\d+)', name)]
    return parts + [name.lower()]


def collect_image_paths(path):
    if os.path.isfile(path):
        if not path.lower().endswith(IMAGE_EXTENSIONS):
            raise ValueError(f'Unsupported image file: {path}')
        return [path]

    if not os.path.isdir(path):
        raise FileNotFoundError(f'Input path not found: {path}')

    paths = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isfile(os.path.join(path, name))
        and name.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if natsorted is not None:
        return natsorted(paths)
    return sorted(paths, key=natural_key)


def load_state_dict(weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ('params', 'params_ema', 'state_dict'):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def strip_or_add_module_prefix(state_dict, model_state):
    state_keys = list(state_dict.keys())
    model_keys = list(model_state.keys())
    if not state_keys or not model_keys:
        return state_dict

    state_has_module = state_keys[0].startswith('module.')
    model_has_module = model_keys[0].startswith('module.')

    if state_has_module and not model_has_module:
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    if model_has_module and not state_has_module:
        return {f'module.{k}': v for k, v in state_dict.items()}
    return state_dict


def patch_sai2e_forward(module):
    def forward_with_cache(self, x):
        feat, scale = self.backbone(x)

        if self.eps != 0:
            safe_sign = torch.where(
                scale >= 0,
                torch.ones_like(scale),
                torch.full_like(scale, -1.0),
            )
            scale = torch.where(scale.abs() < self.eps,
                                safe_sign * self.eps, scale)

        raw_avg = self.rotated_sampler(x, feat)
        res = raw_avg * scale

        self._vis_cache = {
            'res': res.detach(),
            'raw_avg': raw_avg.detach(),
            'scale': scale.detach(),
        }
        return res, scale

    patched = 0
    for child in module.modules():
        if isinstance(child, SAI2E):
            child.forward = MethodType(forward_with_cache, child)
            child._vis_cache = {}
            patched += 1
    if patched == 0:
        raise RuntimeError('No SAI2E module found in network.')


def find_sai2e_cache(module):
    for child in module.modules():
        if isinstance(child, SAI2E) and getattr(child, '_vis_cache', None):
            return child._vis_cache
    raise RuntimeError('SAI2E cache is empty. Did the model forward run?')


def tensor_to_numpy_img(tensor):
    img = torch.clamp(tensor, 0, 1).detach().cpu()
    img = img.squeeze(0).permute(1, 2, 0).numpy()
    return (img * 255.0).round().astype(np.uint8)


COLORMAPS = {
    'jet': cv2.COLORMAP_JET,
    'viridis': cv2.COLORMAP_VIRIDIS,
    'turbo': cv2.COLORMAP_TURBO,
    'plasma': cv2.COLORMAP_PLASMA,
    'magma': cv2.COLORMAP_MAGMA,
}


def feature_to_heatmap(feature, reduce='mean', colormap=cv2.COLORMAP_VIRIDIS):
    feature = feature.detach().float().cpu()
    if feature.ndim == 4:
        feature = feature[0]
    if feature.ndim != 3:
        raise ValueError(f'Expected [C,H,W] or [1,C,H,W], got {feature.shape}')

    if reduce == 'mean':
        feature = feature.mean(dim=0)
    elif reduce == 'max':
        feature = feature.max(dim=0).values
    elif reduce == 'absmean':
        feature = feature.abs().mean(dim=0)
    else:
        raise ValueError(f'Unsupported reduce method: {reduce}')

    feature = feature.numpy()
    min_value = float(feature.min())
    max_value = float(feature.max())
    if max_value - min_value < 1e-12:
        normed = np.zeros_like(feature, dtype=np.uint8)
    else:
        normed = ((feature - min_value) / (max_value - min_value) * 255)
        normed = normed.round().astype(np.uint8)
    return cv2.applyColorMap(normed, colormap)


def save_feature_maps(cache, save_dir, stem, reduce, colormap):
    for name in ('res', 'raw_avg', 'scale'):
        heatmap = feature_to_heatmap(cache[name], reduce=reduce,
                                     colormap=colormap)
        cv2.imwrite(os.path.join(save_dir, f'{stem}_{name}_{reduce}.png'), heatmap)


def build_model(opt_path, weights_path, device, strict):
    opt = parse(opt_path, is_train=False)
    network_opt = deepcopy(opt['network_g'])
    model = define_network(network_opt).to(device)
    state_dict = load_state_dict(weights_path, device)
    state_dict = strip_or_add_module_prefix(state_dict, model.state_dict())
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f'Missing keys: {len(missing)}')
    if unexpected:
        print(f'Unexpected keys: {len(unexpected)}')
    model.eval()
    patch_sai2e_forward(model)
    return model, opt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize SAI2E res/raw_avg/scale features.')
    parser.add_argument('--opt', default='Options/SAIGNet_melol_low_high.yml', help='Path to option YAML file.')
    parser.add_argument('--weights',default='./model_output/MineLOL/best_psnr_25.94_56000.pth', help='Path to model weights.')
    parser.add_argument(
        '--input',
        default='./datasets/yolo/Test_Low/0011.png',
        help='Image file/folder. Defaults to opt.datasets.val.dataroot_lq.')
    parser.add_argument(
        '--save_dir',
        default='visualizations/illu_fea',
        help='Directory for visualization outputs.')
    parser.add_argument(
        '--reduce',
        default='mean',
        choices=['mean', 'max', 'absmean'],
        help='How to reduce channel features to a 2D heatmap.')
    parser.add_argument(
        '--colormap',
        default='viridis',
        choices=sorted(COLORMAPS.keys()),
        help='OpenCV colormap used for feature response heatmaps.')
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Inference device, e.g. cuda, cuda:0, cpu.')
    parser.add_argument(
        '--factor',
        type=int,
        default=4,
        help='Pad images to multiples of this value before inference.')
    parser.add_argument(
        '--save_result',
        action='store_true',
        help='Also save the restored image.')
    parser.add_argument(
        '--non_strict',
        action='store_true',
        help='Load weights with strict=False.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, opt = build_model(args.opt, args.weights, device, not args.non_strict)

    input_path = args.input or opt['datasets']['val']['dataroot_lq']
    image_paths = collect_image_paths(input_path)
    os.makedirs(args.save_dir, exist_ok=True)

    with torch.inference_mode():
        for path in image_paths:
            img = np.float32(utils.load_img(path)) / 255.0
            input_tensor = torch.from_numpy(img).permute(2, 0, 1)
            input_tensor = input_tensor.unsqueeze(0).to(device)

            _, _, h, w = input_tensor.shape
            pad_h = (args.factor - h % args.factor) % args.factor
            pad_w = (args.factor - w % args.factor) % args.factor
            padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), 'reflect')

            restored = model(padded)
            restored = restored[:, :, :h, :w]

            cache = find_sai2e_cache(model)
            cropped_cache = {
                key: value[:, :, :h, :w]
                for key, value in cache.items()
            }

            stem = os.path.splitext(os.path.basename(path))[0]
            save_feature_maps(cropped_cache, args.save_dir, stem, args.reduce,
                              COLORMAPS[args.colormap])

            if args.save_result:
                result = tensor_to_numpy_img(restored)
                utils.save_img(os.path.join(args.save_dir, f'{stem}_result.png'),
                               result)

            print(f'Saved feature maps for {path}')


if __name__ == '__main__':
    main()
