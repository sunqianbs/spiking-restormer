import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


class SNNImageCleanModel(BaseModel):
    """SNN image restoration model."""

    def __init__(self, opt):
        super(SNNImageCleanModel, self).__init__(opt)

        # define network
        self.mixing_flag = False
        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.spike_log_freq = train_opt.get('spike_log_freq', 10)
        self._spike_stat_sum = {}
        self._spike_stat_steps = 0

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)

        loss_dict['l_pix'] = l_pix
        with torch.no_grad():
            res = self.output - self.gt
            loss_dict['mean_abs_res'] = res.abs().mean()
            loss_dict['max_abs_res'] = res.abs().max()

        l_pix.backward()
        if self.opt['train'].get('debug_grad', False):
            params_with_grad = 0
            for p in self.net_g.parameters():
                if p.grad is not None:
                    params_with_grad += 1
            global_norm_raw = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 5.0)
            # compute norm after clipping for verification
            total_sq = None
            for p in self.net_g.parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                sq = g.pow(2).sum()
                total_sq = sq if total_sq is None else total_sq + sq
            global_norm_clipped = torch.sqrt(total_sq) if total_sq is not None else torch.tensor(0.0)
            logger = get_root_logger()
            logger.info(
                f'[grad] global_norm_raw={float(global_norm_raw):.6f}, '
                f'global_norm_clipped={float(global_norm_clipped):.6f}, '
                f'params_with_grad={params_with_grad}')
        elif self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 5.0)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        bare_net = self.get_bare_model(self.net_g)
        if hasattr(bare_net, 'get_spike_stats_and_reset'):
            spike_stats = bare_net.get_spike_stats_and_reset()
            if spike_stats:
                self._spike_stat_steps += 1
                for k, v in spike_stats.items():
                    self._spike_stat_sum[k] = self._spike_stat_sum.get(k, 0.0) + float(v)
                if current_iter % self.spike_log_freq == 0:
                    logger = get_root_logger()
                    avg_stats = {
                        k: (v / max(1, self._spike_stat_steps))
                        for k, v in self._spike_stat_sum.items()
                    }
                    logger.info(
                        "[spike] " + " ".join(
                            f"{k}={avg_stats[k]:.6f}" for k in sorted(avg_stats.keys())
                        )
                    )
                    self._spike_stat_sum = {}
                    self._spike_stat_steps = 0

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def tile_test(self, tile_size, tile_overlap=32):
        """Tile-based inference for validation to reduce peak memory."""
        scale = self.opt.get('scale', 1)
        b, c, h, w = self.lq.size()
        assert b == 1, 'Only batch size 1 is supported for tile test.'

        stride = tile_size - tile_overlap
        if stride <= 0:
            raise ValueError(f'Invalid tile settings: tile_size={tile_size}, tile_overlap={tile_overlap}')

        E = torch.zeros((b, c, h * scale, w * scale), device=self.lq.device)
        W = torch.zeros_like(E)

        h_idx_list = list(range(0, max(h - tile_size, 0) + 1, stride))
        w_idx_list = list(range(0, max(w - tile_size, 0) + 1, stride))
        if len(h_idx_list) == 0 or h_idx_list[-1] != h - tile_size:
            h_idx_list.append(max(h - tile_size, 0))
        if len(w_idx_list) == 0 or w_idx_list[-1] != w - tile_size:
            w_idx_list.append(max(w - tile_size, 0))

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = self.lq[:, :, h_idx:h_idx + tile_size, w_idx:w_idx + tile_size]
                self.nonpad_test(in_patch)
                out_patch = self.output
                out_patch_mask = torch.ones_like(out_patch)

                E[:, :, h_idx * scale:(h_idx + tile_size) * scale, w_idx * scale:(w_idx + tile_size) * scale].add_(out_patch)
                W[:, :, h_idx * scale:(h_idx + tile_size) * scale, w_idx * scale:(w_idx + tile_size) * scale].add_(out_patch_mask)

        self.output = E.div_(W)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        tile_size = self.opt['val'].get('tile_size', 0)
        tile_overlap = self.opt['val'].get('tile_overlap', 32)
        window_size = self.opt['val'].get('window_size', 0)

        if tile_size:
            test = partial(self.tile_test, tile_size, tile_overlap)
        elif window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                save_gt_interval = self.opt['val'].get('save_gt_interval', 1)
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                save_gt = ('gt' in visuals) and (
                    (not self.opt['is_train']) or
                    (isinstance(current_iter, int) and current_iter % save_gt_interval == 0)
                )
                if save_gt:
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
