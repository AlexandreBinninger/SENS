import sys
sys.path.append("/home/amir/projects/sdf_gmm")
import utils.rotation_utils
from custom_types import *
import constants
from options import Options
from utils import train_utils, mcubes_meshing, files_utils, mesh_utils
from models.occ_gmm import OccGen
from models import models_utils
from data_loaders import mesh_datasets


class Inference:

    def split_shape(self, mu: T) -> T:
        b, g, c = mu.shape
        mask = []
        for i in range(b):
            axis = torch.randint(low=0, high=c, size=(1,)).item()
            random_down_top_order = mu[i, :, axis].argsort(dim=-1, descending = torch.randint(low=0, high=2, size=(1,)).item() == 0)
            split_index = g // 4 + torch.randint(g // 2, size=(1,), device=self.device)
            mask.append(random_down_top_order.lt(split_index))  # True- the gaussians we drop
        return torch.stack(mask, dim=0)

    def mix_z(self, gmms, zh):
        with torch.no_grad():
            mu = gmms[0].squeeze(1)[:self.opt.batch_size // 2, :]
            mask = self.split_shape(mu).float().unsqueeze(-1)
            zh_fake = zh[:self.opt.batch_size // 2] * mask + (1 - mask) * zh[self.opt.batch_size // 2:]
        return zh_fake

    def get_occ_fun(self, z: T, gmm: Optional[TS] = None):

        def forward(x: T) -> T:
            nonlocal z
            x = x.unsqueeze(0)
            out = self.model.occ_head(x, z, gmm)[0, :]
            if self.opt.loss_func == LossType.CROSS:
                out = out.softmax(-1)
                out = -1 * out[:, 0] + out[:, 2]
            elif self.opt.loss_func == LossType.IN_OUT:
                out = out / 10
            else:
                out.clamp_(-.2, .2)
            return out
        if z.dim() == 2:
            z = z.unsqueeze(0)
        return forward

    def get_mesh(self, z: T, res: int, gmm: TS, get_time=False) -> Optional[T_Mesh]:

        with torch.no_grad():
            if get_time:
                time_a = self.meshing.occ_meshing(self.get_occ_fun(z, gmm), res=res, get_time=get_time)
                return time_a, time_a
            else:
                mesh = self.meshing.occ_meshing(self.get_occ_fun(z, gmm), res=res)
                return mesh

    def plot_occ(self, z: Union[T, TS], gmms: Optional[List[TS]], prefix: str, res=200, verbose=False,
                  use_item_id: bool = False, fixed_items: Optional[Union[T, List[str]]] = None):
        gmms = gmms[-1]
        if type(fixed_items) is T:
            fixed_items = [f'{fixed_items[item].item():02d}' for item in fixed_items]
        for i in range(len(z)):
            gmm_ = [gmms[j][i].unsqueeze(0) for j in range(len(gmms))]
            mesh = self.get_mesh(z[i], res, [gmm_])
            if use_item_id and fixed_items is not None:
                name = f'_{fixed_items[i]}'
            elif len(z) == 1:
                name = ''
            else:
                name = f'_{i:02d}'
            if mesh is not None:
                files_utils.export_mesh(mesh, f'{self.opt.cp_folder}/occ/{prefix}{name}')
                if gmms is not None:
                    files_utils.export_gmm(gmms, i, f'{self.opt.cp_folder}/gmms/{prefix}{name}')
            if verbose:
                print(f'done {i + 1:d}/{len(z):d}')

    def disentanglement_plot(self, item_a: int, item_b: int, z_in_a, z_in_b, a_inclusive: bool = True,
                             b_inclusive: bool = True):

        def merge_z(z_):
            nonlocal z_in_a, z_in_b, a_inclusive, b_inclusive
            masks = []
            for inds, inclusive in zip((z_in_a, z_in_b), (a_inclusive, b_inclusive)):
                mask_ = torch.zeros(z_.shape[1], dtype=torch.bool)
                mask_[torch.tensor(inds, dtype=torch.long)] = True
                if not inclusive:
                    mask_ = ~mask_
                masks.append(mask_.to(self.device))
            z_a = z_[0][masks[0]]
            z_b = z_[0][~masks[0]]
            if item_b >= 0:
                z_a = torch.cat((z_a, z_[1][masks[1]]), dim=0)
                z_b = torch.cat((z_b, z_[1][~masks[1]]), dim=0)
            return z_a, z_b

        bottom = False
        if item_a < 0 and item_b < 0:
            return
        elif item_a < 0:
            item_a, item_b, z_in_a, z_in_b = item_b, item_a, z_in_b, z_in_a
        suffix = '' if item_b < 0 else f'_{item_b}'
        z_in_a, z_in_b = list(set(z_in_a)), list(set(z_in_b))

        with torch.no_grad():
            if item_b < 0:
                items = torch.tensor([item_a], dtype=torch.int64, device=self.device)
            else:
                items = torch.tensor([item_a, item_b], dtype=torch.int64, device=self.device)
            z_items, z_init, zh, gmms = self.model.get_disentanglement(items)
            if bottom:
                z_in = [z_.unsqueeze(0) for z_ in merge_z(z_init)]
                z_in = [self.model.sdformer.forward_split(self.model.sdformer.forward_upper(z_))[0][0] for z_ in z_in]
            else:
                z_in = merge_z(zh)

            self.plot_sdfs(z_in, None, f'dist_{item_a}{suffix}', verbose=True)

    def compose(self, items: List[int], parts: List[List[int]], inclusive: List[bool]):

        def merge_z() -> T:
            nonlocal inclusive, z_parts, zh
            z_ = []
            for i, (inds, inclusive) in enumerate(zip(z_parts, inclusive)):
                mask_ = torch.zeros(zh.shape[1], dtype=torch.bool)
                mask_[torch.tensor(inds, dtype=torch.long)] = True
                if not inclusive:
                    mask_ = ~mask_
                z_.append(zh[i][mask_])
            z_ = torch.cat(z_, dim=0).unsqueeze_(0)
            return z_
        name = '_'.join([str(item) for item in items])
        z_parts = [list(set(part)) for part in parts]

        with torch.no_grad():
            items = torch.tensor(items, dtype=torch.int64, device=self.device)
            z_items, z_init, zh, gmms = self.model.get_disentanglement(items)
            z_in = merge_z()
            self.plot_sdfs(z_in, None, f'compose_{name}', verbose=True, res=256)

    @models_utils.torch_no_grad
    def get_z_from_file(self, info_path):
        info = files_utils.load_pickle(''.join(info_path))
        items = torch.tensor(list(info.keys()), dtype=torch.int64, device=self.device)
        zh, _, gmms, _ = self.model.get_embeddings(items)
        gmms = gmms[0]
        zh_ = []
        gmms_ = [[] for _ in range(len(gmms))]
        for i in range(len(items)):
            gaussian_inds = torch.tensor(info[items[i].item()], dtype=torch.int64, device=self.device)
            zh_.append(zh[i, gaussian_inds])
            for j in range(len(gmms_)):
                gmms_[j].append(gmms[j][i, :, gaussian_inds])
        zh_ = torch.cat(zh_, dim=0).unsqueeze(0)
        gmms_ = [torch.cat(item, dim=1).unsqueeze(0) for item in gmms_]
        zh, _ = self.model.merge_zh(zh_, [gmms_])
        return zh, zh_, gmms_

    def plot_from_info(self, info_path, res):
        zh, zh_, gmms = self.get_z_from_file(info_path)
        mesh = self.get_mesh(zh[0], res, gmms)
        return mesh

    def get_mesh_interpolation(self, z: T, res: int, mask:TN, alpha: T) -> Optional[T_Mesh]:

        def forward(x: T) -> T:
            nonlocal z, alpha
            x = x.unsqueeze(0)
            out = self.model.occ_head(x, z, mask=mask, alpha=alpha)[0, :]
            out = 2 * out.sigmoid_() - 1
            return out

        mesh = self.meshing.occ_meshing(forward, res=res)
        return mesh

    @staticmethod
    def combine_and_pad(zh_a: T, zh_b: T) -> Tuple[T, TN]:
        if zh_a.shape[1] == zh_b.shape[1]:
            mask = None
        else:
            pad_length = max(zh_a.shape[1], zh_b.shape[1])
            mask = torch.zeros(2, pad_length, device=zh_a.device, dtype=torch.bool)
            padding = torch.zeros(1, abs(zh_a.shape[1] - zh_b.shape[1]), zh_a.shape[-1], device=zh_a.device)
            if zh_a.shape[1] > zh_b.shape[1]:
                mask[1, zh_b.shape[1]:] = True
                zh_b = torch.cat((zh_b, padding), dim=1)
            else:
                mask[0, zh_a.shape[1]: ] = True
                zh_a = torch.cat((zh_a, padding), dim=1)
        return torch.cat((zh_a, zh_b), dim=0), mask

    @staticmethod
    def get_intersection_z(z_a: T, z_b: T) -> T:
        diff = (z_a[0, :, None, :] - z_b[0, None]).abs().sum(-1)
        diff_a = diff.min(1)[0].lt(.1)
        diff_b = diff.min(0)[0].lt(.1)
        if diff_a.shape[0] != diff_b.shape[0]:
            padding = torch.zeros(abs(diff_a.shape[0] - diff_b.shape[0]), device=z_a.device, dtype=torch.bool)
            if diff_a.shape[0] > diff_b.shape[0]:
                diff_b = torch.cat((diff_b, padding))
            else:
                diff_a = torch.cat((diff_a, padding))
        return torch.cat((diff_a, diff_b))

    @models_utils.torch_no_grad
    def get_faces_max_attention(self, mesh: T_Mesh, zh: T, mask: TN = None, alpha: TN = None) -> T:
        coords = mesh[0][mesh[1]].mean(1).unsqueeze(0).to(zh.device)
        attention = self.model.occ_head.forward_attention(coords, zh, mask=mask, alpha=alpha)
        attention = torch.stack(attention, 0).mean(0).mean(-1)
        attention = attention.permute(1, 0, 2).reshape(attention.shape[1], -1)
        attention_max = attention.argmax(-1)
        return attention_max

    @models_utils.torch_no_grad
    def get_attention_sym_dif(self, mesh: T_Mesh, zh: T, mask: T, fixed_z: T, alpha: T) -> T:
        attention_max = self.get_faces_max_attention(mesh, zh, mask, alpha)
        attention_select = fixed_z[attention_max].cpu()
        attention_select = (~attention_select).long()
        return attention_select

    def interpolate_from_files(self, item_a: Union[str, int], item_b: Union[str, int], num_mid: int, res: int,
                               counter: int = 0, logger: Optional[train_utils.Logger] = None):
        zh_a, zh_a_raw, _ = self.get_z_from_file(item_a)
        zh_b, zh_b_raw, _ = self.get_z_from_file(item_b)
        fixed_z = self.get_intersection_z(zh_a_raw, zh_b_raw)
        zh, mask = self.combine_and_pad(zh_a, zh_b)
        if logger is None:
            logger = train_utils.Logger().start(num_mid)
        for i, alpha_ in enumerate(torch.linspace(0, 1, num_mid)):
            alpha = torch.tensor([1., 1.], device=self.device)
            alpha[0] = 1 - alpha_
            alpha[1] = alpha_
            mesh = self.get_mesh_interpolation(zh, res, mask, alpha)
            colors = self.get_attention_sym_dif(mesh, zh, mask, fixed_z, alpha)
            files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/int/trial_{counter + i:03d}")
            files_utils.export_list(colors.tolist(), f"{self.opt.cp_folder}/int/trial_{counter + i:03d}_faces")
            logger.reset_iter()

    @models_utils.torch_no_grad
    def plot_folder(self, *folders, res: int = 220):
        logger = train_utils.Logger()
        for folder in folders:
            paths = files_utils.collect(folder, '.pkl')
            logger.start(len(paths))
            for path in paths:
                name = path[1]
                our_path = f"{self.opt.cp_folder}/from_ui/{name}"
                mesh = self.plot_from_info(path, res)
                if mesh is not None:
                    files_utils.export_mesh(mesh, our_path)
                logger.reset_iter()
            logger.stop()

    def get_samples_by_names(self, names: List[str]) -> T:
        ds = mesh_datasets.CacheDataset(self.opt.dataset_name, self.opt.num_samples, self.opt.data_symmetric)
        return torch.tensor([ds.get_item_by_name(name) for name in names], dtype=torch.int64)

    def get_names_by_samples(self, items):
        ds = mesh_datasets.CacheDataset(self.opt.dataset_name, self.opt.num_samples, self.opt.data_symmetric)
        return [ds.get_name(item) for item in items]

    def get_zh_from_idx(self, items: T):
        zh, _, gmms, __ = self.model.get_embeddings(items.to(self.device))
        zh, attn_b = self.model.merge_zh(zh, gmms)
        return zh, gmms

    @models_utils.torch_no_grad
    def plot(self, prefix: Union[str, int], verbose=False, interpolate: bool = False, res: int = 200, size: int = -1,
             names: Optional[List[str]] = None):
        if size <= 0:
            size = self.opt.batch_size // 2
        if names is not None:
            fixed_items = self.get_samples_by_names(names)
        elif self.model.opt.dataset_size < size:
            fixed_items = torch.arange(self.model.opt.dataset_size)
        else:
            fixed_items = torch.randint(low=0, high=self.opt.dataset_size, size=(size,))
        if names is None:
            names = self.get_names_by_samples(fixed_items)
        # else:
        names = [f'{i:02d}_{name}' for i, name in zip(fixed_items, names)]
        # fixed_items = torch.arange(size) + 100
        # fixed_items = fixed_items.unique()
        # fixed_items = torch.tensor(fixed_items).long().to(self.device)
        if type(prefix) is int:
            prefix = f'{prefix:04d}'
            z, gmms = self.model.interpolate(fixed_items.to(self.device), num_between=8)
            use_item_id = False
        else:
            zh, gmms = self.get_zh_from_idx(fixed_items)
            use_item_id = True
        self.plot_occ(zh, gmms, prefix, verbose=verbose, use_item_id=use_item_id,
                       res=res, fixed_items=names)

    # 1617, 3148, 2175, 529, 1435, 660, 553, 719, 679

    def plot_mix(self):
        fixed_items = torch.randint(low=0, high=self.opt.dataset_size, size=(self.opt.batch_size,))
        with torch.no_grad():
            z, _, gmms = self.model.get_embeddings(fixed_items.to(self.device))
            z = self.mix_z(gmms, z)
            self.plot_occ(z, None, "mix", verbose=True, use_item_id=True, res=200, fixed_items=fixed_items)

    def interpolate_seq(self, num_mid: int, *seq: str, res=200):
        logger = train_utils.Logger().start((len(seq) - 1) * num_mid)
        for i in range(len(seq) - 1):
            self.interpolate_from_files(seq[i], seq[i + 1], num_mid, res, i * num_mid, logger=logger)
        logger.stop()

    def interpolate(self, item_a: Union[str, int], item_b: Union[str, int], num_mid: int, res: int = 200):
        if type(item_a) is str:
            self.interpolate_from_files(item_a, item_b, num_mid, res)
        else:
            zh = self.model.interpolate(item_a, item_b, num_mid)[0]
            for i in range(num_mid):
                mesh = self.get_mesh(zh[i], res)
                files_utils.export_mesh(mesh, f"{self.opt.cp_folder}/interpolate/{item_a}_{item_b}_{i}")
                print(f'done {i + 1:d}/{num_mid}')



    @property
    def device(self):
        return self.opt.device

    def measure_time(self, num_samples: int, *res: int):

        fixed_items = torch.randint(low=0, high=self.opt.dataset_size, size=(num_samples,))
        zh, _, gmms, attn_a = self.model.get_embeddings(fixed_items.to(self.device))
        zh, attn_b = self.model.merge_zh(zh, gmms)
        for res_ in res:
            print(f"\nmeasure {res_:d}")
            times_a, times_b = [], []
            for i in range(len(zh)):
                time_a, time_b = self.get_mesh(zh[i], res_, get_time=True)
                if i > 1:
                    times_a.append(time_a)
                    times_b.append(time_b)
            times_a = torch.tensor(times_a).float()
            times_b = torch.tensor(times_b).float()
            for times in (times_a, times_b):
                print(f"avg: {times.mean()}, std: {times.std()}, min: {times.min()}, , max: {times.max()}")

    @models_utils.torch_no_grad
    def random_samples(self,  nums_sample, res=220):
        logger = train_utils.Logger().start(nums_sample)
        num_batches = nums_sample // self.opt.batch_size + int((nums_sample % self.opt.batch_size) != 0)
        counter = 0
        for batch in range(num_batches):
            if batch == num_batches - 1:
                batch_size = nums_sample - counter
            else:
                batch_size = self.opt.batch_size
            zh, gmms = self.model.random_samples(batch_size)
            for i in range(len(zh)):
                # gmm_ = [gmms[j][i].unsqueeze(0) for j in range(len(gmms))]
                mesh = self.get_mesh(zh[i], res, None)
                pcd = mesh_utils.sample_on_mesh(mesh, 2048, sample_s=mesh_utils.SampleBy.AREAS)[0]
                files_utils.save_np(pcd, f'{self.opt.cp_folder}/gen/pcd_{counter:04d}')
                files_utils.export_mesh(mesh, f'{self.opt.cp_folder}/gen/{counter:04d}')
                logger.reset_iter()
                counter += 1
        logger.stop()

    def plot_from_file(self, item_a: int, item_b: int):
        data = files_utils.load_pickle(f"{self.opt.cp_folder}/compositions/{item_a:d}_{item_b:d}")
        (item_a, gmms_id_a), (item_b, gmms_id_b) = data
        self.disentanglement_plot(item_a, item_b, gmms_id_a, gmms_id_b, b_inclusive=True)

    def plot_from_file_single(self, item_a: int, item_b: int, in_item: int):
        data = files_utils.load_pickle(f"{self.opt.cp_folder}/compositions/{item_a:d}_{item_b:d}")
        (item_a, gmms_id_a) = data[in_item]
        self.disentanglement_plot(item_a, -1, gmms_id_a, [], b_inclusive=True)

    def get_mesh_from_mid(self, gmm, included: T, res: int) -> Tuple[Optional[T_Mesh], TN]:
        if self.mid is None:
            return None, None
        gmm = [elem.to(self.device) for elem in gmm]
        included = included.to(device=self.device)
        mid_ = self.mid[included[:, 0], included[:, 1]].unsqueeze(0)
        zh = self.model.merge_zh(mid_, gmm)[0]
        # zh = self.model_a.merge_zh(self.mid, gmm, mask=mask)[0]
        # zh = zh[included[0]]
        mesh = self.get_mesh(zh[0], res, [gmm])
        colors = self.get_faces_max_attention(mesh, zh).cpu()
        return mesh, colors

    def set_items(self, *items: int):
        items = torch.tensor(items, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            self.mid = self.model.forward_a(items)[0]

    def plot_style_transfer(self, strct_num: Tuple[int, ...], style_num: Tuple[int, ...], res: int = 200, verbose: bool = True):
        items = torch.tensor((*strct_num, *style_num), device=self.device, dtype=torch.int64)
        size = len(strct_num)
        assert size == len(style_num)
        zh, _, gmms, attn_a = self.model.get_embeddings(items.to(self.device))
        zh = zh[size:]
        gmms = [item[:size] for item in gmms]
        zh, _ = self.model.merge_zh(zh, gmms)
        use_item_id = True
        self.plot_occ(zh, gmms, "st", verbose=verbose, use_item_id=use_item_id,
                      res=res, fixed_items=items)


    def plot_single(self, *names: str, res: int = 200):
        gmms = []
        items = []
        paths = []
        included = []
        for name in names:
            paths.append(f'{self.opt.cp_folder}/single_edit/{name}')
            phi, mu, eigen, p, include = files_utils.load_gmm(paths[-1], device=self.device)
            gmms.append([item.unsqueeze(0) for item in (mu, p, phi, eigen)])
            items.append(torch.tensor([int(name.split('_')[0])], device=self.device, dtype=torch.int64))
            included.append(include)
        zh, _, _, attn_a = self.model.forward_a(torch.cat(items, dim=0))
        gmms = [torch.stack(elem) for elem in zip(*gmms)]
        included = torch.stack(included)
        zh = self.model.merge_zh(zh, gmms, mask=~included)
        for z, path, include in zip(zh, paths, included):
            z = z[include]
            mesh = self.get_mesh(z, res)
            files_utils.export_mesh(mesh, path)

    def __init__(self, opt: Options):
        self.opt = opt
        model: Tuple[OccGen, Options] = train_utils.model_lc(opt)
        self.model, self.opt = model
        self.model.eval()
        self.temperature = 1.
        self.mid: Optional[T] = None
        self.gmms: Optional[TN] = None
        self.get_rotation = utils.rotation_utils.rand_bounded_rotation_matrix(100000)
        self.meshing = mcubes_meshing.MarchingCubesMeshing(self.device, max_num_faces=20000)
