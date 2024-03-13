from custom_types import *
from models import models_utils
from options import Options
import math
from torch import distributions


class Decoder(nn.Module):
    def __init__(self, dim_z, dropout: float = 0.2, use_tanh=True, out_ch=1):
        super(Decoder, self).__init__()
        self.dropout = dropout > 0
        dropout_prob = dropout
        self.use_tanh = use_tanh
        in_ch = dim_z
        out_ch = out_ch
        feat_ch = 512

        print("[DeepSDF MLP-9] Dropout: {}; Do_prob: {}; in_ch: {}; hidden_ch: {}".format(self.dropout, dropout_prob,
                                                                                          in_ch, feat_ch))
        if self.dropout is False:
            self.net1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
                nn.ReLU(inplace=True)
            )

            self.net2 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Linear(feat_ch, out_ch)
            )
        else:
            self.net1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
            )

            self.net2 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.Linear(feat_ch, out_ch)
            )

        num_params = sum(p.numel() for p in self.parameters())
        print('[num parameters: {}]'.format(num_params))

    def forward_decoder(self, z, samples, flip_z=False):
        return self.forward(z, samples, flip_z=flip_z)

    # z: [B 131] 128-dim latent code + 3-dim xyz
    def forward(self, z, samples: Optional[T], flip_z=False):
        if samples is not None:
            if z.dim() == 2:
                embeddings = z.unsqueeze(1).expand(*samples.shape[:-1], -1)
            else:
                embeddings = z
            if flip_z:
                samples = samples * 10. / 9.
                x_ = samples[:, :, 0]
                z_ = -samples[:, :, 1]
                y_ = samples[:, :, 2]
                samples = torch.stack((x_, z_, y_), -1)
            z = torch.cat((embeddings, samples), dim=2)
        in1 = z
        out1 = self.net1(in1)
        in2 = torch.cat([out1, in1], dim=-1)
        out2 = self.net2(in2)
        if self.use_tanh:
            out2 = torch.tanh(out2)
        return out2


class DeepSdf(models_utils.Model):

    def get_random_embeddings(self, num_items):
        weights = self.z.weight.clone().detach()
        mean = weights.mean(0)
        weights = weights - mean[None, :]
        cov = torch.einsum('nd,nc->dc', weights, weights) / (weights.shape[0] - 1)
        dist = distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)
        z_init = dist.sample((num_items,))
        return z_init

    def forward_decoder(self, z, x, flip_z=False):
        return self.decoder(z, x, flip_z=flip_z)

    def forward(self, item, x: T):
        z = self.z(item)
        return self.decoder(z, x), z

    def __init__(self, opt: Options):
        super(DeepSdf, self).__init__()
        self.decoder = Decoder(opt.dim_z + 3)
        self.z = nn.Embedding(opt.dataset_size, opt.dim_z)
        torch.nn.init.normal_(
            self.z.weight.data,
            0.0,
            1. / math.sqrt(opt.dim_z),
        )
