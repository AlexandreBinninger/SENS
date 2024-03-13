import torch

from custom_types import *
from chamferdist import ChamferDistance
from emd_imp import emd_module
from models import models_utils
from utils import files_utils, mesh_utils


# from PyTorchEMD.emd import earth_mover_distance


# def mesh_acc(points: T, mesh: T_Mesh, th: float = .9) -> float:
#     distances: T = vs_cp(points, mesh)[1]
#     k = int_b(distances.shape[0] * th)
#     return distances.kthvalue(k).item()


class Evaluator:

    @staticmethod
    def print_evaluation(emds: T, chamfers: T, mesh_acc: T):
        emds = emds.sort()[0]
        chamfers = chamfers.sort()[0]
        mesh_acc = mesh_acc.sort()[0]
        median_index = chamfers.shape[0] // 2
        print(f"{chamfers.mean() * 1000:.3f} & {chamfers[median_index] * 1000:.3f} &"
              f" {emds.mean() * 1000:.2f} & {emds[median_index] * 1000:.2f} & {mesh_acc.mean() * 1000:.3f}")
        print(f"chamfer mean: {chamfers.mean() * 1000:.3f} chamfer median: {chamfers[median_index] * 1000:.3f}\n"
              f"emd mean:     {emds.mean() * 1000:.2f}     emd median:     {emds[median_index] * 1000:.2f}\n"
              f"acc mean:     {mesh_acc.mean() * 1000:.3f} acc median:     {mesh_acc[median_index] * 1000:.3f}")

    def save(self, path: str):
        self.empty_buffer()
        emds = torch.cat(self.emds)
        chamfers = torch.cat(self.chamfers)
        mesh_acc = torch.cat(self.mesh_acc)
        self.print_evaluation(emds, chamfers, mesh_acc)
        files_utils.save_pickle({"emd": emds, "chamfer": chamfers, "mesh_acc": mesh_acc}, path)

    def empty_buffer(self):
        if len(self.buffers[0]) > 0:
            pc_a, pc_b = torch.stack(self.buffers[0], dim=0).to(self.device), torch.stack(self.buffers[1], dim=0).to(self.device)
            self(pc_a, pc_b)
        self.buffers = [], []

    def fill_buffer(self, mesh_a: T_Mesh, mesh_b: T_Mesh):
        scale = (mesh_b[0].max(0)[0] - mesh_b[0].min(0)[0]).norm(2)
        samples_a = mesh_utils.sample_on_mesh(mesh_a, self.num_samples, sample_s=mesh_utils.SampleBy.AREAS)[0]
        samples_b = mesh_utils.sample_on_mesh(mesh_b, self.num_samples, sample_s=mesh_utils.SampleBy.AREAS)[0]
        self.buffers[0].append(samples_a / scale)
        self.buffers[1].append(samples_b / scale)
        if len(self.buffers[0]) == self.batch_size:
            self.empty_buffer()

    @staticmethod
    def prepare_emd(pc_a, pc_b):
        if pc_a.shape[1] > 1024:
            order = (torch.rand(pc_a.shape[1]).argsort()[:1024]).to(pc_a.device)
            arange = torch.arange(pc_a.shape[0], device=pc_a.device)
            pc_a, pc_b = pc_a[arange, order], pc_b[arange, order]
        return pc_a, pc_b

    @models_utils.torch_no_grad
    def __call__(self, pred_pc: T, gt_pc: T):
        chamfer_distance_a, chamfer_distance_b = self.chamfer_evaluator(pred_pc, gt_pc, bidirectional=True)
        chamfer_distance = (chamfer_distance_a ** 2).mean(1) + (chamfer_distance_b ** 2).mean(1)
        # emd = earth_mover_distance(*self.prepare_emd(pred_pc, gt_pc), transpose=False)
        dis, _ = self.emd_evaluator(pred_pc, gt_pc, 0.05, 3000)
        emd = np.sqrt(dis.cpu()).mean(-1)
        k = int(chamfer_distance_a.shape[1] * self.mesh_acc_th)
        mesh_acc, _ = chamfer_distance_a.kthvalue(k, dim=1)
        self.chamfers.append(chamfer_distance.cpu())
        self.emds.append(emd)
        self.mesh_acc.append(mesh_acc.cpu())
        # for pc, mesh in zip(pred_pc, gt_meshes):
        #     self.mesh_acc.append(mesh_acc(pc, mesh))

    def __init__(self, num_samples: int, batch_size: int, device: D, mesh_acc_th: float = .9):
        self.mesh_acc_th = mesh_acc_th
        self.device = device
        self.chamfer_evaluator = ChamferDistance()
        self.emd_evaluator = emd_module.emdModule()
        self.emds = []
        self.chamfers = []
        self.mesh_acc = []
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.buffers = [], []


class EvaluatorIoU:

    def print_evaluation(self):
        print(self.experiment_name)
        for arr, name in zip((self.iou_surface_a, self.iou_surface_b, self.iou_random),
                             ('surface', 'surface', 'random')):
            values = torch.tensor(arr)
            mean, median = values.mean(), values.median()
            print(f"{name} mean: {mean}   {name} median: {median}")

    def __call__(self, pred: T, labels: T):
        labels = labels.lt(0)
        pred = pred.lt(0)
        iou = (pred * labels).sum(-1) / (pred + labels).sum(-1)
        self.iou_surface_a.append(iou[0].item())
        self.iou_surface_b.append(iou[1].item())
        self.iou_random.append(iou[2].item())

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.iou_surface_a = []
        self.iou_surface_b = []
        self.iou_random = []


class EvaluatorParts:

    @staticmethod
    def print_evaluation(log: Dict[str, T]):
        mean_part = torch.cat((log["iou_parts_near_b"], log["iou_parts_random"], log["iou_parts_inside"])).mean()
        mean_all = torch.cat((log["iou_all_near_b"], log["iou_all_random"], log["iou_all_inside"])).mean()
        print(f"& {mean_part:.3f} & {mean_all:.3f}")
        for key, item in log.items():
            print(f"{key}: {item.mean():.3f} {key}: {item.median():.3f}")

    def save(self, path: str):
        log = {}
        for key, item in self.log.items():
            log[key] = torch.tensor(item)
        return log

    def save(self, path: str):
        log = self.get_log()
        files_utils.save_pickle(log, path)
        self.print_evaluation(log)

    def iou_occ(self, occ_a: T, occ_b: T, keys: Tuple[str, ...]):
        occ_a, occ_b = occ_a.view(len(keys), -1), occ_b.view(len(keys), -1)
        iou = (occ_a * occ_b).sum(1).float() / (occ_b + occ_b).sum(1).float()
        for i, key in enumerate(keys):
            if not torch.isnan(iou[i]):
                self.log[key].append(iou[i])

    def evaluate_part(self, i: int, occ: T, occ_all: T, coords_labels: T, gmm_labels: T):
        occ_all[~coords_labels.eq(i)] = 0
        self.iou_occ(occ_all, occ, ("iou_parts_near_a", "iou_parts_near_b", "iou_parts_random", "iou_parts_inside"))
        # if coords_part_labels.shape[0] != 0:
        #     self.iou_parts.append((coords_part_labels.sum().float() / coords_part_labels.shape[0]).item())
        # coords_part_labels = coords_labels.eq(i)

    def __call__(self, occ_all: T, split_out: Dict[int, T], supports: T, gmm_labels: T):
        coords_labels = gmm_labels[supports.argmax(-1)]
        union: TN = None
        for i, occ in split_out.items():
            self.evaluate_part(i, occ, occ_all.clone(), coords_labels, gmm_labels)
            if union is None:
                union = occ
            else:
                union = union + occ
        self.iou_occ(union, occ_all, ("iou_all_near_a", "iou_all_near_b", "iou_all_random", "iou_all_inside"))

    def __init__(self):
        self.log = {"iou_parts_near_a": [], "iou_parts_near_b": [], "iou_parts_random": [], "iou_parts_inside": [],
                    "iou_all_near_a": [], "iou_all_near_b": [], "iou_all_random": [], "iou_all_inside": []}


if __name__ == '__main__':
    # evaluator = Evaluator()
    x, y = torch.rand(5, 2 ** 10, 3).cuda(), torch.rand(5, 2 ** 10, 3).cuda()
    # out_a = earth_mover_distance(x, y, transpose=False)
    out_b, _ = emd_module.emdModule()(x, y, 0.05, 3000)
    exit(0)

