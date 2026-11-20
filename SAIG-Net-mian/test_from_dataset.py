import numpy as np
import os
import argparse
import re
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    import utils
except ModuleNotFoundError as exc:
    if exc.name != 'utils':
        raise
    from Enhancement import utils

try:
    from natsort import natsorted
except ImportError:
    natsorted = None
from skimage import img_as_ubyte

from basicsr.models import create_model
from basicsr.utils.options import parse

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def natural_key(path):
    name = os.path.basename(path)
    parts = [int(part) if part.isdigit() else part.lower()
             for part in re.split(r'(\d+)', name)]
    return parts + [name.lower()]


def collect_image_paths(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'Image folder not found: {folder}')
    paths = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
        and name.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if natsorted is not None:
        return natsorted(paths)
    return sorted(paths, key=natural_key)


def image_paths_by_stem(paths, folder_name):
    paths_by_stem = {}
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in paths_by_stem:
            raise ValueError(
                f'Duplicate image stem "{stem}" found in {folder_name}.')
        paths_by_stem[stem] = path
    return paths_by_stem


def get_paired_image_paths(input_dir, target_dir):
    input_paths = collect_image_paths(input_dir)
    target_paths = collect_image_paths(target_dir)
    if len(input_paths) != len(target_paths):
        raise ValueError(
            'Input and target folders have different numbers of images: '
            f'{len(input_paths)}, {len(target_paths)}.')

    input_paths_by_stem = image_paths_by_stem(input_paths, input_dir)
    paired_paths = []
    for target_path in target_paths:
        stem = os.path.splitext(os.path.basename(target_path))[0]
        input_path = input_paths_by_stem.get(stem)
        if input_path is None:
            raise ValueError(f'{stem} is not in input paths: {input_dir}.')
        paired_paths.append((input_path, target_path))
    return paired_paths


def build_lpips_metric(enabled, net='alex'):
    if not enabled:
        return None

    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            'LPIPS evaluation requires the `lpips` package. '
            'Install it first or run without `--lpips`.'
        ) from exc

    metric = lpips.LPIPS(net=net).cuda()
    metric.eval()
    return metric


def compute_lpips(metric, target, restored):
    if metric is None:
        return None

    target_tensor = torch.from_numpy(
        np.ascontiguousarray(target)
    ).permute(2, 0, 1).unsqueeze(0).float().cuda()
    restored_tensor = torch.from_numpy(
        np.ascontiguousarray(restored)
    ).permute(2, 0, 1).unsqueeze(0).float().cuda()

    return metric(restored_tensor, target_tensor, normalize=True).item()

def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)

parser = argparse.ArgumentParser(
    description='SAIGNet')

parser.add_argument('--result_dir', default='./results',
                    type=str, help='Directory for results')
parser.add_argument('--output_dir', default='',
                    type=str, help='Directory for output')
parser.add_argument(
    '--opt', type=str, default='Options/SAIGNet_melol_low_high.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='./model_output/MineLOL/psnr_25.94_56000.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='melol_low_high', type=str,
                    help='Test Dataset') 
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble to obtain better results')
parser.add_argument('--lpips', dest='lpips', action='store_true',
                    help='Compute LPIPS together with PSNR/SSIM (enabled by default)')
parser.add_argument('--no_lpips', dest='lpips', action='store_false',
                    help='Disable LPIPS computation')
parser.add_argument('--lpips_net', default='alex', choices=['alex', 'vgg', 'squeeze'],
                    type=str, help='LPIPS backbone when --lpips is enabled')
parser.set_defaults(lpips=True)

args = parser.parse_args()

gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

####### Load yaml #######
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    opt['num_gpu'] = 0
print(f'Inference device: {device}')


x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################


model_restoration = create_model(opt).net_g

checkpoint = torch.load(weights, map_location=device, weights_only=False)

try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.to(device)
if device.type == 'cuda':
    model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()
has_gt = bool(opt['datasets']['val'].get('dataroot_gt'))
if not has_gt and args.GT_mean:
    parser.error('--GT_mean requires datasets.val.dataroot_gt')
lpips_metric = build_lpips_metric(args.lpips and has_gt, args.lpips_net)
if lpips_metric is not None:
    print(f"LPIPS enabled with backbone: {args.lpips_net}")

factor = 4
dataset = args.dataset
config = os.path.splitext(os.path.basename(args.opt))[0]
checkpoint_name = os.path.splitext(os.path.basename(args.weights))[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
output_dir = args.output_dir
# stx()
os.makedirs(result_dir, exist_ok=True)
if args.output_dir != '':
    os.makedirs(output_dir, exist_ok=True)

psnr = []
ssim = []
lpips_scores = []
if dataset in ['SID', 'SMID', 'LOL_Blur']:
    os.makedirs(result_dir_input, exist_ok=True)
    os.makedirs(result_dir_gt, exist_ok=True)
    if dataset == 'SID':
        from basicsr.data.SID_image_dataset import Dataset_SIDImage as Dataset
    elif dataset == 'SMID':
        from basicsr.data.SMID_image_dataset import Dataset_SMIDImage as Dataset
    elif dataset == 'LOL_Blur':
        from basicsr.data.LOL_Blur_dataset import Dataset_LOLBlurImage as Dataset
    else:
        from basicsr.data.SDSD_image_dataset import Dataset_SDSDImage as Dataset
    opt = opt['datasets']['val']
    opt['phase'] = 'test'
    if opt.get('scale') is None:
        opt['scale'] = 1
    if '~' in opt['dataroot_gt']:
        opt['dataroot_gt'] = os.path.expanduser('~') + opt['dataroot_gt'][1:]
    if '~' in opt['dataroot_lq']:
        opt['dataroot_lq'] = os.path.expanduser('~') + opt['dataroot_lq'][1:]
    dataset = Dataset(opt)
    print(f'test dataset length: {len(dataset)}')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_batch['lq']
            input_save = data_batch['lq'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
            target = data_batch['gt'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            # Padding in case images are not multiples of 4
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if args.self_ensemble:
                restored = self_ensemble(input_, model_restoration)
            else:
                restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]

            restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                # This test setting is the same as KinD, LLFlow, and recent diffusion models
                # Please refer to Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py)
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            current_psnr = utils.PSNR(target, restored)
            current_ssim = utils.calculate_ssim(
                img_as_ubyte(target), img_as_ubyte(restored))
            current_lpips = compute_lpips(lpips_metric, target, restored)

            psnr.append(current_psnr)
            ssim.append(current_ssim)
            if current_lpips is not None:
                lpips_scores.append(current_lpips)
            type_id = os.path.dirname(inp_path).split('/')[-1]
            os.makedirs(os.path.join(result_dir, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_input, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_gt, type_id), exist_ok=True)
            utils.save_img((os.path.join(result_dir, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
            utils.save_img((os.path.join(result_dir_input, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(input_save))
            utils.save_img((os.path.join(result_dir_gt, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(target))
else:

    input_dir = opt['datasets']['val']['dataroot_lq']
    target_dir = opt['datasets']['val'].get('dataroot_gt')
    paired_paths = (get_paired_image_paths(input_dir, target_dir)
                    if target_dir else [(path, None) for path in collect_image_paths(input_dir)])

    with torch.inference_mode():
        for inp_path, tar_path in tqdm(paired_paths, total=len(paired_paths)):

            if device.type == 'cuda':
                torch.cuda.empty_cache()

            img = np.float32(utils.load_img(inp_path)) / 255.
            target = np.float32(utils.load_img(tar_path)) / 255. if tar_path else None

            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).to(device)

            # Padding in case images are not multiples of 4
            b, c, h, w = input_.shape
            H, W = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if h < 3000 and w < 3000:
                if args.self_ensemble:
                    restored = self_ensemble(input_, model_restoration)
                else:
                    restored = model_restoration(input_)
            else:
                # split and test
                input_1 = input_[:, :, :, 1::2]
                input_2 = input_[:, :, :, 0::2]
                if args.self_ensemble:
                    restored_1 = self_ensemble(input_1, model_restoration)
                    restored_2 = self_ensemble(input_2, model_restoration)
                else:
                    restored_1 = model_restoration(input_1)
                    restored_2 = model_restoration(input_2)
                restored = torch.zeros_like(input_)
                restored[:, :, :, 1::2] = restored_1
                restored[:, :, :, 0::2] = restored_2

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]

            restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            if args.GT_mean:
                # This test setting is the same as KinD, LLFlow, and recent diffusion models
                # Please refer to Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py)
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            log_parts = [f'id: {os.path.splitext(os.path.basename(inp_path))[0]}']
            if target is not None:
                current_psnr = utils.PSNR(target, restored)
                current_ssim = utils.calculate_ssim(
                    img_as_ubyte(target), img_as_ubyte(restored))
                current_lpips = compute_lpips(lpips_metric, target, restored)
                log_parts.extend([f'PSNR: {current_psnr}', f'SSIM: {current_ssim}'])
                if current_lpips is not None:
                    log_parts.append(f'LPIPS: {current_lpips}')
                psnr.append(current_psnr)
                ssim.append(current_ssim)
                if current_lpips is not None:
                    lpips_scores.append(current_lpips)
            print('\n', *log_parts)
            if output_dir != '':
                utils.save_img((os.path.join(output_dir, os.path.splitext(
                    os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
            else:
                utils.save_img((os.path.join(result_dir, os.path.splitext(
                    os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))

if psnr:
    print("PSNR: %f " % np.mean(psnr))
    print("SSIM: %f " % np.mean(ssim))
if lpips_scores:
    print("LPIPS: %f " % (np.mean(np.array(lpips_scores))))
