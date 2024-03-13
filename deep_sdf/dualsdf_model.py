from custom_types import *
from deep_sdf.deepsdf_model import Decoder
import numpy as np
import torch
import torch.nn as nn
from models import models_utils
import options
from torch import distributions


def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix


def kld(mu, logvar):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    kld = torch.mean(kld)
    return kld


class SDFFun(nn.Module):
    def __init__(self):
        super(SDFFun, self).__init__()
        self.return_idx = False
        self.smooth = True
        self.smooth_factor = 100
        print('[SdfSphere] return idx: {}; smooth: {}'.format(self.return_idx, self.smooth))

    # assume we use Sphere primitive for everything
    # parameters: radius[r], center[xyz]
    def prim_sphere_batched_smooth(self, x, p):
        device = x.device
        x = x.unsqueeze(-2)  # B N 1 3
        p = p.unsqueeze(-3)  # B 1 M 4
        logr = p[:, :, :, 0]
        d = torch.sqrt(torch.sum((x - p[:, :, :, 1:4]) ** 2, dim=-1)) - torch.exp(logr)  # B N M
        if self.return_idx:
            d, loc = torch.min(d, dim=-1, keepdim=True)
            return d, loc
        else:
            if self.smooth:
                d = bsmin(d, dim=-1, k=self.smooth_factor, keepdim=True)
            else:
                d, _ = torch.min(d, dim=-1, keepdim=True)
            return d

    # a: [B M 4];
    # x: [B N 3]; x, y, z \in [-0.5, 0.5]
    def forward(self, a, x):
        a = a.reshape(a.size(0), -1, 4)
        out = self.prim_sphere_batched_smooth(x, a)
        return out


class VADLogVar(nn.Module):
    def __init__(self, N, dim):
        super(VADLogVar, self).__init__()
        self.N = N
        self.dim = dim
        self.weight_mu = nn.Parameter(torch.Tensor(N, dim))
        self.weight_logvar = nn.Parameter(torch.Tensor(N, dim))
        self.reset_parameters()
        print('[VADLogVar Embedding] #entries: {}; #dims: {}'.format(N, dim))

    def reset_parameters(self):
        mu_init_std = 1.0 / np.sqrt(self.dim)
        torch.nn.init.normal_(
            self.weight_mu.data,
            0.0,
            mu_init_std,
        )

        logvar_init_std = 1.0 / np.sqrt(self.dim)
        torch.nn.init.normal_(
            self.weight_logvar.data,
            0,
            logvar_init_std,
        )

    def forward(self, idx, num_augment_pts=None):
        mu = self.weight_mu[idx]
        logvar = self.weight_logvar[idx]
        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            batch_latent = mu + eps * std
            eps_aug = torch.randn(std.size(0), num_augment_pts, std.size(1), device=std.device)
            batch_latent_aug = mu.unsqueeze(1) + eps_aug * std.unsqueeze(1)
            return {'latent_code': batch_latent, 'latent_code_augment': batch_latent_aug, 'mu': mu, 'logvar': logvar,
                    'std': std}
        else:
            print('[VADLogVar Embedding] Test mode forward')
            batch_latent = mu
            return {'latent_code': batch_latent, 'mu': mu, 'logvar': logvar, 'std': std}


def clamped_l1_correct(pred_dist, gt_dist, trunc=0.1):
    gt_dist = gt_dist.view(gt_dist.shape[0], -1)
    pred_dist = pred_dist.view(pred_dist.shape[0], -1)
    pred_dist_lower = torch.clamp(pred_dist, None, trunc)
    pred_dist_upper = torch.clamp(pred_dist, -trunc, None)
    pos_trunced_mask = (gt_dist >= trunc)
    neg_trunced_mask = (gt_dist <= -trunc)
    valid_mask = ~(pos_trunced_mask | neg_trunced_mask)
    loss_valid = torch.sum(torch.abs(pred_dist - gt_dist) * valid_mask.float(), dim=-1)
    loss_lower = torch.sum((trunc - pred_dist_lower) * pos_trunced_mask.float(), dim=-1)
    loss_upper = torch.sum((pred_dist_upper + trunc) * neg_trunced_mask.float(), dim=-1)
    loss = (loss_lower + loss_upper + loss_valid) / pred_dist.size(1)
    return loss


# L2 loss on the outside, encourage inside to < 0.0
def onesided_l2(pred_dist, gt_dist):
    gt_dist = gt_dist.view(gt_dist.shape[0], -1)
    pred_dist = pred_dist.view(pred_dist.shape[0], -1)
    valid_mask = (gt_dist >= 0.0).float()
    num_valid = torch.sum(valid_mask, dim=-1)
    num_inside = valid_mask[0].numel() - num_valid
    loss_valid = torch.sum((gt_dist - pred_dist) ** 2 * valid_mask, dim=-1) / (num_valid + 1e-8)
    loss_inside = torch.sum(torch.clamp(pred_dist, 0.0, None) * (1.0 - valid_mask), dim=-1) / (num_inside + 1e-8)
    loss = loss_valid + loss_inside
    return loss


class DualSdfDecoders(models_utils.Model):

    def forward(self, z_dict, points_coarse, points_fine):
        attr = self.prim_attr_net(z_dict['latent_code'].detach(), None)
        out_a = self.prim_sdf_fun(attr, points_coarse)
        out_b = self.deepsdf_net(z_dict['latent_code_augment'], points_fine)
        return out_a, out_b, attr

    def __init__(self):
        super(DualSdfDecoders, self).__init__()
        self.deepsdf_net = Decoder(131, 0.2, True, 1)
        self.prim_attr_net = Decoder(128, 0, False, 1024)
        self.prim_sdf_fun = SDFFun()


class DualSdf(models_utils.Model):

    def forward_decoder(self, z, samples, flip_z=False):
        return self.decoders.deepsdf_net.forward(z, samples, flip_z=flip_z)

    def forward(self, items, points_coarse, points_fine):
        z_dict = self.z(items, points_fine.shape[1])
        out_a, out_b, attr = self.decoders(z_dict, points_coarse, points_fine)
        return out_a, out_b, attr, z_dict

    def get_random_embeddings(self, num_items):
        weights = self.z.weight_mu.data.clone().detach()
        mean = weights.mean(0)
        weights = weights - mean[None, :]
        cov = torch.einsum('nd,nc->dc', weights, weights) / (weights.shape[0] - 1)
        dist = distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)
        z_init = dist.sample((num_items,))
        return z_init

    def __init__(self, opt: options.Options):
        super(DualSdf, self).__init__()
        self.z = VADLogVar(opt.dataset_size, 128)
        self.decoders = DualSdfDecoders()


def load_model(path: str, device: D) -> Decoder:

    def remove_prefix(name: str) -> str:
         return '.'.join(name.split('.')[1:])

    model = Decoder(131)
    weights = torch.load(path)['trainer_state_dict']
    weights = {remove_prefix(key): item for key, item in weights.items() if 'deepsdf_net' in key}
    model.load_state_dict(weights)
    model = model.to(device)
    return model



def main():
    model = DualSdf(options.Options(dataset_size=10))
    # model = load_model("./pretrained/dual_sdf_airplanes.pth", CPU).eval()
    pts_a = torch.rand(5, 2048, 3)
    pts_b = torch.rand(5, 2048, 3)
    items = torch.arange(5)
    out_a, out_b, attr, z_dict = model(items, pts_a, pts_b)
    # out_a = model(x).flatten()
    # out_b = model(y).flatten()
    # diff = (out_a - out_b).norm(p=2)
    # print(diff)


if __name__ == '__main__':
    main()



