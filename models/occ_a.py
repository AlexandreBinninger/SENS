from options import Options
from models import models_utils, transformer, mlp_models, deep_sdf
import constants
from custom_types import *
from torch import distributions
import math


class ProcessEmbSimple(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        super(ProcessEmbSimple, self).__init__()
        upsample = [
            nn.Linear(emb_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(True),
            models_utils.View(-1, 2, hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.ReLU(True),
            models_utils.View(-1, 4, hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            models_utils.View(-1, 1, 8, hidden_dim)
        ]
        self.net_up = nn.Sequential(*upsample)

    def forward(self, *args):
        return self.net_up(args[0])


def dot(x, y, dim=3):
    return torch.sum(x * y, dim=dim)


def remove_projection(v_1, v_2):
    proj = (dot(v_1, v_2) / dot(v_2, v_2))
    return v_1 - proj[:, :, :, None] * v_2


def split_gm(splitted):
    raw_base = []
    for i in range(constants.DIM):
        u = splitted[i]
        for j in range(i):
            u = remove_projection(u, raw_base[j])
        raw_base.append(u)
    p = torch.stack(raw_base, dim=3)
    p = p / torch.norm(p, p=2, dim=4)[:, :, :, :, None]  # + self.noise[None, None, :, :]
    # eigenvalues
    eigen = splitted[constants.DIM] ** 2 + constants.EPSILON
    sigma_det = eigen[:, :, :, 0]
    for i in range(1, constants.DIM):
        sigma_det = sigma_det * eigen[:, :, :, i]
    mu = splitted[constants.DIM + 1]
    phi = splitted[constants.DIM + 2].squeeze(3)
    return mu, p, sigma_det, phi, eigen


class GMCast(nn.Module):

    def __init__(self, hidden_dim: int, z_dim: int):
        super(GMCast, self).__init__()
        projection_dim = constants.DIM ** 2 + 2 * constants.DIM + 1 + z_dim
        self.mlp = models_utils.MLP((hidden_dim, hidden_dim // 2, projection_dim), dropout=0.1)
        self.split_shape = tuple((constants.DIM + 2) * [constants.DIM] + [1, z_dim])

    def forward(self, x):
        x = self.mlp(x)
        return split_gm(torch.split(x, self.split_shape, dim=3))


class ParamWrapper(nn.Module):

    def forward(self):
        return self.param

    def __init__(self, x, requires_grad=True):
        super(ParamWrapper, self).__init__()
        self.param = nn.Parameter(x, requires_grad)


class SingleGMM(nn.Module):

    def forward(self):
        return [split_gm(torch.split(param(), self.split_shape, dim=3)) for param in self.params], [z() for z in self.z]

    def __init__(self, opt: Options):
        super(SingleGMM, self).__init__()
        projection_dim = constants.DIM ** 2 + 2 * constants.DIM + 1
        params = [(2 * torch.rand(1, 1, 32, projection_dim) - 1)] + [
            (2 * torch.rand(1, 32 * 4 ** i, 4, projection_dim) - 1) for i in range(opt.num_splits)]
        z = [torch.randn(1, 1, 32, opt.dim_zh)] + [torch.randn(1, 32 * 4 ** i, 4, opt.dim_zh)
                                                   for i in range(opt.num_splits)]
        self.params = nn.ModuleList([ParamWrapper(param) for param in params])
        self.z = nn.ModuleList([ParamWrapper(param) for param in z])
        self.split_shape = tuple((constants.DIM + 2) * [constants.DIM] + [1])


class PointGMM(nn.Module):

    def __init__(self, opt: Options):
        super(PointGMM, self).__init__()
        self.process_layer = ProcessEmbSimple(opt.dim_z, opt.dim_h)
        self.projector = GMCast(opt.dim_h, opt.dim_zh)
        self.mid_projector = nn.ModuleList([GMCast(opt.dim_h, opt.dim_zh) for _ in range(opt.num_splits)])
        if opt.attentive:
            self.attention = nn.ModuleList([models_utils.GMAttend(opt.dim_h) for _ in range(opt.num_splits)])
        else:
            self.attention = [models_utils.Dummy() for _ in range(opt.num_splits)]
        self.mlp_split = nn.ModuleList([models_utils.MLP((opt.dim_h, opt.dim_h * 2, opt.dim_h * 4), dropout=0.1)
                                        for _ in range(opt.num_splits)])

    def forward(self, x: T) -> List[Tuple[T, ...]]:
        gms = []
        raw_gm = self.process_layer(x)
        for i in range(len(self.attention)):
            gms.append(self.mid_projector[i](raw_gm))
            raw_gm = self.gm_split(raw_gm, self.attention[i], self.mlp_split[i])
        gms.append(self.projector(raw_gm))
        return gms

    @staticmethod
    def gm_split(x: T, attention: nn.Module, mlp: nn.Module) -> T:
        b_size, grand_parents, parents, dim = x.shape
        x = attention(x)
        out = mlp(x).view(b_size, grand_parents, parents, -1, dim)
        return out.view(b_size, grand_parents * parents, -1, dim)


class GmmTransformer(nn.Module):

    def forward_bottom(self, x):
        return self.l1(x).view(-1, self.bottom_width, self.embed_dim)

    def forward_upper(self, x):
        return self.transformer(x)

    def forward(self, x):
        x = self.forward_bottom(x)
        x = self.forward_upper(x)
        return x

    def __init__(self, opt: Options, act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm):
        super(GmmTransformer, self).__init__()
        self.bottom_width = opt.num_gaussians
        self.embed_dim = opt.dim_h
        self.l1 = nn.Linear(opt.dim_z, self.bottom_width * opt.dim_h)
        self.transformer = transformer.Transformer(opt.dim_h, opt.num_heads, opt.num_layers, act=act,
                                                   norm_layer=norm_layer)


class SdfHead(nn.Module):

    def forward(self, x: T, zh: T, mask: Optional[T] = None) -> T:
        pos_encoder = self.pos_encoder(x)
        x = torch.cat((x, pos_encoder), dim=2)
        z = self.sdf_transformer(x, zh, mask)
        out = self.sdf_mlp(x, z)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

    def __init__(self, opt: Options):
        super(SdfHead, self).__init__()
        self.pos_encoder = mlp_models.SineLayer(constants.DIM, opt.pos_dim, is_first=True)
        self.sdf_transformer = transformer.Transformer(opt.pos_dim + constants.DIM, opt.num_heads, opt.num_layers,
                                                       dim_ref=opt.dim_h)
        self.sdf_mlp = deep_sdf.DeepSDF(opt, dims=opt.head_sdf_size * [512],
                                        latent_in=(opt.head_sdf_size // 2 + opt.head_sdf_size % 2,))


class OccFormer(models_utils.Model):

    def forward_bottom(self, x):
        z_bottom = self.embedding_transformer.forward_bottom(x)
        return z_bottom

    def forward_upper(self, x):
        x = self.embedding_transformer.forward_upper(x)
        return x

    def forward_split(self, x):
        raw_gmm = self.to_gmm(x).unsqueeze(1)
        zh = self.to_zh(x)
        gmms = split_gm(torch.split(raw_gmm, self.split_shape, dim=3))
        return zh, gmms

    @staticmethod
    def apply_gmm_affine(gmms: TS, affine: T):
        mu, p, sigma_det, phi, eigen = gmms
        if affine.dim() == 2:
            affine = affine.unsqueeze(0).expand(mu.shape[0], *affine.shape)
        mu_r = torch.einsum('bad, bpnd->bpna', affine, mu)
        p_r = torch.einsum('bad, bpndc->bpnac', affine, p)
        return mu_r, p_r, sigma_det, phi, eigen

    def concat_gmm(self, gmm_a, gmm_b):
        out = []
        for element in zip(gmm_a, gmm_b):
            out.append(torch.cat(element, dim=2))
        return out

    def forward(self, z_init):
        zs = self.embedding_transformer(z_init)
        zh, gmms = self.forward_split(zs)
        if self.reflect is not None:
            gmms_r = self.apply_gmm_affine(gmms, self.reflect)
            gmms = self.concat_gmm(gmms, gmms_r)
            zh = torch.cat((zh, zh), dim=1)
        return zh, gmms

    @staticmethod
    def get_reflection(reflect_axes: Tuple[bool, ...]):
        reflect = torch.eye(constants.DIM)
        for i in range(constants.DIM):
            if reflect_axes[i]:
                reflect[i, i] = -1
        return reflect

    def __init__(self, opt: Options):
        super(OccFormer, self).__init__()
        if sum(opt.symmetric) > 0:
            reflect = self.get_reflection(opt.symmetric)
            self.register_buffer("reflect", reflect)
        else:
            self.reflect = None

        self.split_shape = tuple((constants.DIM + 2) * [constants.DIM] + [1])
        self.embedding_transformer = GmmTransformer(opt)
        self.to_gmm = nn.Linear(opt.dim_h, sum(self.split_shape))
        self.to_zh = nn.Linear(opt.dim_h, opt.dim_h - sum(self.split_shape))


class OccGen(models_utils.Model):

    def get_z(self, item: T):
        return self.z(item)

    @staticmethod
    def interpolate_(z, num_between: Optional[int] = None):
        if num_between is None:
            num_between = z.shape[0]
        alphas = torch.linspace(0, 1, num_between, device=z.device)
        while alphas.dim() != z.dim():
            alphas.unsqueeze_(-1)
        z_between = alphas * z[1:2] + (- alphas + 1) * z[:1]
        return z_between

    def interpolate_higher(self, z: T, num_between: Optional[int] = None):
        z_between = self.interpolate_(z, num_between)
        zh, gmms = self.occ_former.forward_split(self.occ_former.forward_upper(z_between))
        return zh, gmms

    def interpolate(self, item_a: int, item_b: int, num_between: int):
        items = torch.tensor((item_a, item_b), dtype=torch.int64, device=self.device)
        z = self.get_z(items)
        z_between = self.interpolate_(z, num_between)
        zh, gmms = self.occ_former(z_between)
        return zh, gmms

    def get_disentanglement(self, items: T):
        z_a = self.get_z(items)
        z_b = self.occ_former.forward_bottom(z_a)
        zh, gmms = self.occ_former.forward_split(self.occ_former.forward_upper(z_b))
        return z_a, z_b, zh, gmms

    def get_embeddings(self, item: T):
        z = self.get_z(item)
        zh, gmms = self.occ_former(z)
        return zh, z, gmms

    @staticmethod
    def merge_zh(zh, gmms) -> T:
        mu, p, _, phi, eigen = [item.squeeze(1) for item in gmms]
        p = p.reshape(*p.shape[:2], -1)
        z_gmm = torch.cat((mu, p, phi.unsqueeze(-1), eigen), dim=2).detach()
        zh_ = torch.cat((zh, z_gmm), dim=2)
        return zh_

    def forward_b(self, x, zh, gmms, mask: Optional[T] = None) -> T:
        zh = self.merge_zh(zh, gmms)
        return self.occ_head(x, zh, mask)

    def forward_a(self, item: T):
        zh, z, gmms = self.get_embeddings(item)
        return zh, z, gmms

    def forward(self, x, item: T) -> Tuple[T, T, TS, T]:
        zh, z, gmms = self.forward_a(item)
        return self.forward_b(x, zh, gmms), z, gmms, zh

    def random_samples(self, num_items: int):
        weights = self.z.weight.clone().detach()
        mean = weights.mean(0)
        weights = weights - mean[None, :]
        cov = torch.einsum('nd,nc->dc', weights, weights) / (weights.shape[0] - 1)
        dist = distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)
        z_init = dist.sample((num_items,))
        zh, gmms = self.occ_former(z_init)
        return zh, gmms

    def __init__(self, opt: Options):
        super(OccGen, self).__init__()
        self.device = opt.device
        self.z = nn.Embedding(opt.dataset_size, opt.dim_z)
        torch.nn.init.normal_(
            self.z.weight.data,
            0.0,
            1. / math.sqrt(opt.dim_z),
        )
        self.occ_former = OccFormer(opt)
        self.occ_head = SdfHead(opt)


def main():
    opt = Options(dataset_size=50)
    # model = SdFormerGen(opt_)
    # x = torch.rand(4, 512)
    # out = model(x)
    model = OccFormer(opt)
    x = torch.rand(4, opt.dim_z)
    out = model(x)
    # model = OccGen(opt).to(CUDA(0))
    # x = torch.rand(24, 1024, 3, device=CUDA(0))
    # items = torch.arange(24, device=CUDA(0))
    # out, z_mean, log_sigma, gmms = model(x, items)
    # print(out.shape)
    # print(z_mean.shape)
    # print(gmms[0].shape)


if __name__ == '__main__':
    main()



