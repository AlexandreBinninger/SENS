import torch

from custom_types import *
import constants
from data_loaders import sketch_dataset
from options import SketchOptions, Options
from utils import train_utils, files_utils
from models.sketch_model import Sketch2Spaghetti
from ui_sketch import occ_inference
from models import models_utils

vertices_cube = np.array([[1, 1, 1],
                     [1, 1, -1],
                     [1, -1, 1],
                     [1, -1, -1],
                     [-1, 1, 1],
                     [-1, 1, -1],
                     [-1, -1, 1],
                     [-1, -1, -1]])

faces_cube =  np.array([[0, 1, 2],
                  [1, 3, 2],
                  [2, 3, 6],
                  [3, 7, 6],
                  [0, 4, 1],
                  [1, 4, 5],
                  [4, 6, 5],
                  [5, 6, 7],
                  [0, 2, 4],
                  [2, 6, 4],
                  [1, 5, 3],
                  [3, 5, 7]])

class SketchInference:

    @property
    def device(self):
        return self.opt.device

    def prepare_real_sketches(self, sketch):
        sketch = torch.from_numpy(sketch)
        sketch = sketch.float().mean(-1) / 255.
        sketch = sketch.unsqueeze(0).unsqueeze(0)
        sketch = nnf.interpolate(sketch, 256,  mode='bicubic', align_corners=True).gt(0.9).float()
        return sketch.to(self.device)

    def get_mesh(self, res = 128, thresh_mask = 0.00):
        out_z, gmms = self.spaghetti.model.occ_former.forward_mid([self.zh])
        out_z = self.spaghetti.model.merge_zh_step_a(out_z, gmms)
        mask = self.out_cls.ge(thresh_mask)
        out_z, _ = self.spaghetti.model.affine_transformer.forward_with_attention(out_z[0][mask].unsqueeze(0))
        out_path = f'{self.opt.cp_folder}/inference/tmp.obj'
        self.mesh = self.spaghetti.get_mesh(out_z[0], res, None)
        if self.mesh is None:
            return gmms[0], (vertices_cube, faces_cube)
        self.gmm = gmms
        return self.gmm[0], self.mesh

    @models_utils.torch_no_grad
    def sketch2mesh(self, sketch: ARRAY, _zh = None, get_zh = False):
        sketch = self.prepare_real_sketches(sketch)
        self.zh, out_cls, self.z_mid = self.model(sketch, True)
        self.out_cls = out_cls.squeeze().sigmoid()
        if get_zh:
            return *(self.get_mesh()), self.zh
        return self.get_mesh()
    
    def sketch2mesh_partial(self, sketch: ARRAY, selected = Optional[ARRAY], _zh = Optional[T], get_zh = False):
        if selected is None or _zh is None:
            print("called partial sketch2mesh without specifying partial embeddings")
            return self.sketch2mesh(sketch, _zh = _zh, get_zh = get_zh)
        selected = torch.tensor(selected, device=self.device, dtype=torch.bool)
        _zh = _zh.to(self.device)
        sketch = self.prepare_real_sketches(sketch)
        self.zh, out_cls, self.z_mid = self.model(sketch, True)
        self.zh = self.zh.to(self.device)
        self.zh[:, ~selected, :] = _zh[:, ~selected, :]
        self.out_cls = out_cls.squeeze().sigmoid()
        if get_zh:
            return *(self.get_mesh()), self.zh
        return self.get_mesh()

    @models_utils.torch_no_grad
    def rebuild_mesh(self, selected):
        if self.z_mid is None:
            return None, None
        selected = torch.tensor(selected, device=self.device, dtype=torch.bool)
        mask = 1 - selected.float()
        z_mid = self.z_mid * mask[None, :, None]
        out_z, z_mid = self.model.refine(z_mid, True)
        self.z_mid = self.z_mid * mask[None, :, None] + (1 - mask)[None, :, None] * z_mid
        self.zh = self.zh * mask[None, :, None] + (1 - mask)[None, :, None] * out_z
        return self.get_mesh()

    def __init__(self, opt: SketchOptions):
        self.spaghetti = occ_inference.Inference(Options(tag=opt.spaghetti_tag).load())
        self.opt = opt
        model: Tuple[Sketch2Spaghetti, SketchOptions] = train_utils.model_lc(opt)
        self.model, self.opt = model
        self.model.eval()
        self.mesh = None
        self.gmm = None
        self.zh: TN = None
        self.z_mid: TN = None
        self.out_cls: TN = None


