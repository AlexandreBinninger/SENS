from custom_types import *


def clamped_l1(pred_dist, gt_dist, trunc=0.1):
    b = pred_dist.shape[0]
    pred_dist, gt_dist = pred_dist.view(b, -1), gt_dist.view(b, -1)
    pred_dist_trunc = torch.clamp(pred_dist, -trunc, trunc)
    gt_dist_trunc = torch.clamp(gt_dist, -trunc, trunc)
    loss = nnf.l1_loss(pred_dist_trunc, gt_dist_trunc)
    return loss


def sdf_hinge_loss(sdf_predict: T, sdf_gt: T, temperature: float = 1, delta: float = 1) -> T:
    sdf_gt = torch.sign(sdf_gt) * ((sdf_gt.abs() ** temperature) * (delta ** (1 - temperature))).clamp_min_(delta)
    # sdf_gt[sdf_gt.gt(0)] = 1
    # sdf_gt[sdf_gt.lt(0)] = -1
    sdf_predict: T = sdf_predict.clamp(-delta, delta)
    loss = nnf.l1_loss(sdf_predict, sdf_gt.view(sdf_gt.shape[0], -1))
    return loss


def sdf_cross_loss(sdf_predict: T, sdf_gt: T, *_) -> T:
    sdf_gt = sdf_gt.flatten()
    sdf_predict = sdf_predict.view(-1, sdf_predict.shape[-1])
    labels = sdf_gt.eq(0).long()
    labels[sdf_gt.gt(0)] = 2
    loss = nnf.cross_entropy(sdf_predict, labels)
    return loss


def occupancy_bce(sdf_predict: T, sdf_gt: T, ignore: Optional[T] = None, *args) -> T:
    labels = sdf_gt.flatten().gt(0).float()
    sdf_predict = sdf_predict.flatten()
    if ignore is not None:
        ignore = (~ignore).flatten().float()
    loss = nnf.binary_cross_entropy_with_logits(sdf_predict, labels, weight=ignore)
    return loss


def dkl_loss(z_mean: T, log_sigma: T):
    loss = -0.5 * torch.sum(1 + log_sigma - z_mean.pow(2) - log_sigma.exp(), dim=-1)
    return loss.mean()


def reg_z_loss(z: T) -> T:
    norms = z.norm(2, 1)
    loss = norms.mean()
    return loss