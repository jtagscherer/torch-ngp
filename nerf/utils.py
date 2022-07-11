import os
import glob
import tqdm
import math
import random
import warnings
import tensorboardX

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver

from styletransfer.RAIN import Net as RainNet
from styletransfer.model import StyleNeRFpp
from styletransfer.utils import load_style_image, get_content_loss, get_style_loss

patch_sampling = False


def enablePatchSampling(enable):
    global patch_sampling
    patch_sampling = enable


def getPatchSampling():
    return patch_sampling


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, random_patches=False, ray_resolution=None,
             previous_inds=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)
        if not random_patches:
            if error_map is None:
                inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
                inds = inds.expand([B, N])
            else:
                # weighted sample on a low-reso grid
                inds_coarse = torch.multinomial(error_map.to(device), N,
                                                replacement=False)  # [B, N], but in [0, 128*128)

                # map to the original resolution with random perturb.
                inds_x, inds_y = inds_coarse // 128, inds_coarse % 128  # `//` will throw a warning in torch 1.10... anyway.
                sx, sy = H / 128, W / 128
                inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
                inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
                inds = inds_x * W + inds_y

                results['inds_coarse'] = inds_coarse  # need this when updating error_map

        else:
            # Patch-wise training - Random choose one fixed pixel per region (region is random size and random position)
            total_inds = torch.arange(H * W).reshape(H, W)
            patch_H, patch_W = 67, 81
            num_region_H, num_region_W = H // patch_H, W // patch_W  # 16, 24(Family, Francis, Horse), #8, 12(Truck, PG)

            if previous_inds is not None:
                inds = previous_inds
            else:
                region_size_v = np.random.randint(num_region_H // 2, num_region_H + 1)
                region_size_u = np.random.randint(num_region_W // 3, num_region_W + 1)
                region_position_v = np.random.randint(H - patch_H * region_size_v + region_size_v)
                region_position_u = np.random.randint(W - patch_W * region_size_u + region_size_u)
                inds = total_inds[region_position_v::region_size_v][:patch_H][:, region_position_u::region_size_u][:,
                       :patch_W]

            results['previous_inds'] = inds

            if ray_resolution is not None:
                inx_w = np.array([i for i in range(81) if i % ray_resolution != 0])
                inx_h = np.array([i for i in range(67) if i % ray_resolution != 0])
                # inx = np.flip(np.transpose([np.tile(inx_w, len(inx_h)), np.repeat(inx_h, len(inx_w))]), 1)
                inds = inds[np.repeat(inx_h, len(inx_w)), np.tile(inx_w, len(inx_h))]

            inds = inds.reshape(-1)
            inds = inds.expand([B, inds.size(0)])
            inds = inds.to(device)

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1, 2, 0).squeeze()
        x = x.detach().cpu().numpy()

    x = x.astype(np.float32)

    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    # plt.show()


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys),
                                                  len(zs)).detach().cpu().numpy()  # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    # print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    # print(u.shape, u.max(), u.min(), np.percentile(u, 50))

    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean(np.power(preds - truths, 2)))

        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class Trainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 lr_scheduler=None,  # scheduler
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=2,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
                 ):

        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        self.style_model = StyleNeRFpp().to(device)
        self.style_image = load_style_image()

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        # clip loss prepare
        if opt.rand_pose >= 0:  # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text])  # only support one text prompt now...

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def train_step(self, data, patch_data=None):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        images = data['images']  # [B, N, 3/4]
        B, N, C = images.shape

        # train in srgb color space
        if C == 4:
            # train with random background color if using alpha mixing
            # bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            # bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            bg_color = torch.rand_like(images[..., :3])  # [N, 3], pixel-wise random.
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            bg_color = None
            gt_rgb = images

        style_training_start_step = 1000
        depth_steps = 5

        if self.global_step == style_training_start_step:
            enablePatchSampling(True)

        if self.global_step > style_training_start_step:
            # Freeze NeRF if not frozen yet
            if self.global_step == style_training_start_step + 1:
                for param in self.model.sigma_net.parameters():
                    param.requires_grad = False
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)

            if 'images' in data:
                outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=None, perturb=True,
                                            force_all_rays=True,
                                            **vars(self.opt))

                prediction_depth = outputs['depth'].detach()

                minimal_depth = torch.min(prediction_depth)
                maximal_depth = torch.max(prediction_depth)
                depth_per_step = (maximal_depth - minimal_depth) / (depth_steps * 1.0)

                loss = None

                for depth_level in range(depth_steps):
                    resolution = pow(depth_level + 2, 2)

                    inx_w = np.array([i for i in range(81) if i % resolution != 0])
                    inx_h = np.array([i for i in range(67) if i % resolution != 0])
                    prediction_width = len(inx_w)
                    prediction_height = len(inx_h)

                    ground_truth = gt_rgb.reshape(67, 81, 3)[np.repeat(inx_h, len(inx_w)),
                                   np.tile(inx_w, len(inx_h)),
                                   :].reshape(1, prediction_width * prediction_height, 3)

                    # DEBUG
                    full_patch = outputs['image'].detach()
                    full_gt = ground_truth.detach()

                    rays = get_rays(data['poses'], data['intrinsics'], data['H'], data['W'], 67 * 81,
                                    random_patches=True,
                                    ray_resolution=resolution, previous_inds=data['previous_inds'])
                    rays_o = rays['rays_o']  # [B, N, 3]
                    rays_d = rays['rays_d']  # [B, N, 3]
                    outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=None, perturb=True,
                                                force_all_rays=True,
                                                **vars(self.opt))

                    prediction = outputs['image']
                    prediction_depth = outputs['depth']
                    prediction_mask = torch.where(
                        (prediction_depth > (minimal_depth + depth_level * depth_per_step)) & (
                                    prediction_depth < (minimal_depth + (depth_level + 1) * depth_per_step)),
                        torch.ones_like(prediction_depth), torch.zeros_like(prediction_depth)).reshape(prediction_height, prediction_width, 1)

                    # Erode the prediction map analogously to Hoellein et al.
                    '''k = torch.ones(1, 1, 3, 3).type_as(prediction_mask)
                    erosion_mask = torch.nn.functional.conv2d(prediction_mask, k, padding=(1, 1)) / 3 ** 2
                    erosion_mask = torch.clamp(erosion_mask, 0, 1)
                    prediction_mask = prediction_mask * (erosion_mask == 1)'''

                    content_feat = self.style_model.get_content_feat(
                        ground_truth.reshape(prediction_height, prediction_width, 3).permute(2, 0,
                                                                                             1).contiguous().unsqueeze(
                            0), mask=prediction_mask)
                    output_content_feat = self.style_model.get_content_feat(
                        prediction.reshape(prediction_height, prediction_width, 3).permute(2, 0,
                                                                                           1).contiguous().unsqueeze(
                            0), mask=prediction_mask)

                    output_style_feats, output_style_feat_mean_std = self.style_model.get_style_feat(
                        prediction.reshape(prediction_height, prediction_width, 3).permute(2, 0,
                                                                                           1).contiguous().unsqueeze(
                            0), mask=prediction_mask)
                    style_feats, style_feat_mean_std = self.style_model.get_style_feat(
                        self.style_image.cuda().unsqueeze(0))

                    if self.global_step < style_training_start_step + 10:
                        print(
                            f'{self.global_step}: Depth level: {depth_level}, Resolution: {resolution} ({prediction_width} x {prediction_height})')

                        pred_image = prediction.detach()
                        pred_image = pred_image.reshape(prediction_height, prediction_width, 3).permute(2, 0,
                                                                                                        1).contiguous()
                        torch_vis_2d(pred_image)
                        plt.savefig(f'/tmp/nerfout/{self.global_step}_{depth_level}_pred.png')

                        depth_image = prediction_depth.detach()
                        depth_image = depth_image.reshape(prediction_height, prediction_width, 1).permute(2, 0,
                                                                                                          1).contiguous()
                        torch_vis_2d(depth_image)
                        plt.savefig(f'/tmp/nerfout/{self.global_step}_{depth_level}_depth.png')

                        gt_image = ground_truth.detach()
                        gt_image = gt_image.reshape(prediction_height, prediction_width, 3).permute(2, 0,
                                                                                                    1).contiguous()
                        torch_vis_2d(gt_image)
                        plt.savefig(f'/tmp/nerfout/{self.global_step}_{depth_level}_gt.png')

                        mask_image = prediction_mask.detach()
                        mask_image = mask_image.permute(2, 0, 1).contiguous()
                        torch_vis_2d(mask_image)
                        plt.savefig(f'/tmp/nerfout/{self.global_step}_{depth_level}_mask.png')

                    content_loss = get_content_loss(content_feat, output_content_feat)
                    nerf_loss = self.criterion(prediction, ground_truth).mean()
                    style_loss = get_style_loss(style_feat_mean_std, output_style_feat_mean_std)

                    if self.global_step <= style_training_start_step + 1500:
                        current_loss = content_loss + nerf_loss
                    else:
                        current_loss = content_loss + style_loss

                    if loss is None:
                        loss = current_loss
                    else:
                        loss += current_loss

                '''def mask_prediction(prediction, depth):
                    median_depth = torch.median(prediction_depth)
                    reshaped_prediction_depth = torch.repeat_interleave(torch.unsqueeze(depth, axis=2), 3, axis=2)
                    return torch.where(reshaped_prediction_depth > median_depth, prediction, torch.zeros_like(prediction))'''

                return prediction, ground_truth, loss

        # if there is no gt image, we train with CLIP loss.
        if 'images' not in data:
            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=None, perturb=True, force_all_rays=True,
                                        **vars(self.opt))
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()

            # [debug] uncomment to plot the images used in train_step
            # torch_vis_2d(pred_rgb[0])

            loss = self.clip_loss(pred_rgb)

            return pred_rgb, None, loss

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, **vars(self.opt))

        pred_rgb = outputs['image']

        loss = self.criterion(pred_rgb, gt_rgb).mean(-1)  # [B, N, 3] --> [B, N]

        # update error_map
        if self.error_map is not None:
            index = data['index']  # [B]
            inds = data['inds_coarse']  # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index]  # [B, H * W]

            # [debug] uncomment to save and visualize error map
            # if self.global_step % 1001 == 0:
            #     tmp = error_map[0].view(128, 128).cpu().numpy()
            #     print(f'[write error map] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
            #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            #     cv2.imwrite(os.path.join(self.workspace, f'{self.global_step}.jpg'), (tmp * 255).astype(np.uint8))

            error = loss.detach().to(error_map.device)  # [B, N], already in [0, 1]

            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        if patch_data is not None:
            style_prediction = \
                self.model.render(patch_data['rays_o'], patch_data['rays_d'], staged=False, bg_color=None, perturb=True,
                                  force_all_rays=True,
                                  **vars(self.opt))['image']
            ground_truth = patch_data['images']
            output_style_feats, output_style_feat_mean_std = self.style_model.get_style_feat(
                style_prediction.reshape(67, 81, 3).permute(2, 0, 1).contiguous().unsqueeze(0))
            style_feats, style_feat_mean_std = self.style_model.get_style_feat(self.style_image.cuda().unsqueeze(0))

            content_feat = self.style_model.get_content_feat(
                ground_truth.reshape(67, 81, 3).permute(2, 0, 1).contiguous().unsqueeze(0))
            output_content_feat = self.style_model.get_content_feat(
                style_prediction.reshape(67, 81, 3).permute(2, 0, 1).contiguous().unsqueeze(0))
            style_loss = get_style_loss(style_feat_mean_std, output_style_feat_mean_std)
            content_loss = get_content_loss(content_feat, output_content_feat)
            loss = content_loss + 0.01 * style_loss

        loss = loss.mean()

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        images = data['images']  # [B, H, W, 3/4]
        B, H, W, C = images.shape

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth

    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:],
                                               resolution=resolution, threshold=threshold, query_func=query_func)

        mesh = trimesh.Trimesh(vertices, triangles, process=False)  # important, process=True leads to seg fault...
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map

        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        with torch.no_grad():

            # update grid
            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)

                path = os.path.join(save_path, f'{i:04d}.png')
                path_depth = os.path.join(save_path, f'{i:04d}_depth.png')

                # self.log(f"[INFO] saving test image to {path}")

                cv2.imwrite(path,
                            cv2.cvtColor((preds[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(path_depth, (preds_depth[0].detach().cpu().numpy() * 255).astype(np.uint8))

                pbar.update(loader.batch_size)

        self.log(f"==> Finished Test.")

    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(train_loader)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            '''if self.global_step == 5001:
                        for param in self.model.sigma_net.parameters():
                            param.requires_grad = False'''
            patch_data = None
            try:
                data = next(loader)
                '''if self.global_step > 5000:
                    enablePatchSampling(True)
                    patch_data = next(loader)
                    enablePatchSampling(False)
                else:
                    patch_data = None'''
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # mark untrained grid
            if self.global_step == 0:
                self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)
                self.error_map = train_loader._data.error_map

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % 16 == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data, patch_data=patch_data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return outputs

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                preds, preds_depth = self.test_step(data, bg_color=bg_color, perturb=spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3,
                                                                                                  1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        outputs = {
            'image': preds[0].detach().cpu().numpy(),
            'depth': preds_depth[0].detach().cpu().numpy(),
        }

        return outputs

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")
        print(f"Start Training Epoch {self.epoch}")
        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # update grid
        if self.model.cuda_ray:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:

            enablePatchSampling(True)
            patch_data = next(iter(loader))
            enablePatchSampling(False)

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data, patch_data=patch_data)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            # update grid
            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in
                                  range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in
                                        range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in
                                   range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation',
                                             f'{self.name}_{self.epoch:04d}_{self.local_step:04d}.png')
                    save_path_depth = os.path.join(self.workspace, 'validation',
                                                   f'{self.name}_{self.epoch:04d}_{self.local_step:04d}_depth.png')
                    # save_path_gt = os.path.join(self.workspace, 'validation', f'{self.name}_{self.epoch:04d}_{self.local_step:04d}_gt.png')

                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, cv2.cvtColor((preds[0].detach().cpu().numpy() * 255).astype(np.uint8),
                                                        cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, (preds_depth[0].detach().cpu().numpy() * 255).astype(np.uint8))
                    # cv2.imwrite(save_path_gt, cv2.cvtColor((truths[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)  # if max mode, use -result
            else:
                self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
