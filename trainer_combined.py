from pydoc import cli
from custom_types import *
import constants
from data_loaders import combined_dataset
from options import SketchOptions, Options
from utils import train_utils, files_utils
from models.sketch_model import Sketch2Spaghetti
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from ui_sketch import occ_inference
from models import models_utils
import matplotlib.pyplot as plt
import argparse


class TrainerCombined:

    def get_mesh(self, zh):
        out_z, gmms = self.spaghetti.model.occ_former.forward_mid(zh.unsqueeze(0).unsqueeze(0))
        out_z = self.spaghetti.model.merge_zh_step_a(out_z, gmms)
        out_z, _ = self.spaghetti.model.affine_transformer.forward_with_attention(out_z[0][:].unsqueeze(0))
        mesh = self.spaghetti.get_mesh(out_z[0], 256, None)
        if mesh is None:
            return None, None
        gmm = gmms
        return gmm[0], mesh

    def validation_iter(self, data: TS):
        with torch.no_grad():
            weight_partial_loss, sketches_full, sketches, masks, zh  = self.prepare_data(data)
            out_zh_full, _, z_mid = self.model(sketches_full, True)
            loss_full = nnf.l1_loss(out_zh_full, zh, reduction="mean")
            return loss_full

    def validation_loss(self, epoch):
        self.logger.start(len(self.val_dl), tag=self.opt.tag + ' validation')
        loss_val = torch.zeros((1)).to(self.device)

        self.val_dl.dataset.set_augmentation(False)
        for data in self.val_dl:
            loss_val += (self.validation_iter(data) / len(self.val_dl))
            self.logger.stash_iter("loss_validation", loss_val)
            self.logger.reset_iter()
        self.writer.add_scalar("Loss/validation", loss_val.item(), epoch)
        self.val_dl.dataset.set_augmentation(True)
        return self.logger.stop()

    def between_epochs(self, train_dict, epoch):
        val_dict = self.validation_loss(epoch)
        if val_dict['loss_validation'] < self.best_scores_validation:
            self.model.save()
            files_utils.copy_file(f'{self.opt.cp_folder}/model', f'{self.opt.cp_folder}/model_val', force=True)
            self.best_scores_validation = val_dict['loss_validation']
            self.improved = True
        if epoch > self.offset and (epoch - self.offset) % self.opt.lr_decay_every == 0:
            self.scheduler.step()
        if (epoch + 1) % 500 == 0 and self.improved:
            files_utils.copy_file(f'{self.opt.cp_folder}/model_val', f'{self.opt.cp_folder}/model_{epoch:06d}')
            self.improved = False

    def prepare_data(self, data: TS) -> TS:
        return tuple(map(lambda x: x.to(self.device), data))

    def refinement_iter(self, z_mid, zh):
        mask = torch.cuda.FloatTensor(z_mid.shape[:-1]).uniform_().gt(self.opt.max_refinement / self.opt.num_gaussians)
        z_mid = z_mid * mask[:, :, None]
        out = self.model.refine(z_mid)  
        loss = nnf.l1_loss(out, zh, reduction='none')
        loss = loss * (1 - mask / 2.)[:, :, None]
        loss = loss.mean()
        return loss
    
    def get_spaghetti_occ_val(self, out_zh, pc):
        out_z, gmms = self.spaghetti.model.occ_former.forward_mid([out_zh])
        out_z = self.spaghetti.model.merge_zh_step_a(out_z, gmms)
        out_z, _ = self.spaghetti.model.affine_transformer.forward_with_attention(out_z)
        out = self.spaghetti.model.occ_head(pc, out_z)
        return out

    def get_ref_mesh(self, zh):
        out_z, gmms = self.spaghetti.model.occ_former.forward_mid(zh.unsqueeze(0).unsqueeze(0))
        out_z = self.spaghetti.model.merge_zh_step_a(out_z, gmms)
        out_z, _ = self.spaghetti.model.affine_transformer.forward_with_attention(out_z[0].unsqueeze(0))
        mesh = self.spaghetti.get_mesh(out_z[0], 300, None)
        return mesh


    def train_iter(self, data: TS, is_train: bool):
        self.optimizer.zero_grad()
        weight_partial_loss, sketches_full, sketches, masks, zh = self.prepare_data(data)
        out_zh_part, out_cls = self.model(sketches)
        out_zh_full, _, z_mid = self.model(sketches_full, True)

        len_partial = weight_partial_loss.sum() + 1e-5
        loss_part = nnf.l1_loss(weight_partial_loss * out_zh_part * masks, weight_partial_loss * zh * masks, reduction="mean")/len_partial
        loss_full = nnf.l1_loss(out_zh_full, zh, reduction="mean")
        loss_cls = self.weight_cls * nnf.binary_cross_entropy_with_logits(out_cls, masks, weight=weight_partial_loss, reduction="mean")/len_partial
        loss = loss_part + loss_full+ loss_cls
        self.logger.stash_iter('loss_zh_full', loss_full, 'loss_zh_part', loss_part, 'loss_cls', loss_cls)

        if self.opt.refinement:
            loss_refinement = self.refinement_iter(z_mid, zh)
            self.logger.stash_iter('loss_refinement', loss_refinement)
            loss += loss_refinement
        loss.backward()
        self.optimizer.step()
        self.warm_up_scheduler.step()
        return loss

    def train_epoch(self, epoch: int, is_train: bool):
        self.model.train(is_train)
        self.logger.start(len(self.dl), tag=self.opt.tag + ' train' if is_train else ' val')
        loss_train = torch.zeros((1)).to(self.device)
        for data in self.dl:
            loss_train += (self.train_iter(data, is_train) / len(self.dl))
            self.logger.reset_iter()
        self.writer.add_scalar("Loss/train", loss_train.item(), epoch)
        return self.logger.stop()

    def reset_dl(self) -> DataLoader:
        from copy import deepcopy
        opt_prosketch = deepcopy(self.opt)
        opt_prosketch.data_tag="ProSketch-3Dchair"
        ds = combined_dataset.CombinedDs(self.opt, self.opt, opt_prosketch)

        dataset_size = len(ds)
        indices = list(range(dataset_size))
        split = int(np.floor(self.opt.val_frac * dataset_size))
        shuffle = not constants.DEBUG
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        files_utils.save_np(val_indices, f'{self.opt.cp_folder}/validation')
        
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(ds, batch_size=self.opt.batch_size, pin_memory=True,
                        num_workers=0 if constants.DEBUG else 12, drop_last=True,
                                                sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(ds, batch_size=self.opt.batch_size,pin_memory=True,
                        num_workers=0 if constants.DEBUG else 12, drop_last=True,
                                                sampler=valid_sampler)
        return train_loader, validation_loader

    def train(self):
        for i in range(self.opt.epochs):
            train_dict = self.train_epoch(i, True)
            self.between_epochs(train_dict, i + 1)

    @property
    def device(self):
        return self.opt.device

    def prepare_real_sketches(self, folder):
        paths = files_utils.collect(f'{constants.DATA_ROOT}/{folder}/', '.png')
        images = [files_utils.load_image(''.join(path)) for path in paths]
        images = [torch.from_numpy(image) for image in images]
        images = [image.float().mean(-1) / 255. for image in images]
        images = [image.unsqueeze(0).unsqueeze(0) for image in images]
        images = [nnf.interpolate(image, 256,  mode='bicubic', align_corners=True).gt(0.9).float() for image in images]
        images = torch.cat(images)
        names = [item[1] for item in paths]
        return images.to(self.device), names

    def set_spaghetti(self, opt):
        self.spaghetti = occ_inference.Inference(opt)
        print("spahetti for occ:", opt.tag)

    def __init__(self, opt: SketchOptions):
        self.spaghetti: Optional[occ_inference.Inference] = None
        self.opt = opt
        self.dl, self.val_dl = self.reset_dl()
        self.offset = opt.warm_up // len(self.dl)
        model: Tuple[Sketch2Spaghetti, SketchOptions] = train_utils.model_lc(opt)
        self.model, self.opt = model
        self.optimizer = Optimizer(self.model.parameters(), lr=1e-7)
        self.warm_up_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1e2, total_epoch=opt.warm_up)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opt.lr_decay)
        self.logger = train_utils.Logger()
        self.best_scores_zh_full = 100
        self.best_scores_validation = 100
        self.plot_scale = 1.
        self.improved = False
        self.writer = SummaryWriter(f'{opt.cp_folder}/runs')
        self.weight_cls = opt.weight_cls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer for combined dataset (NPR + clipasso + ProSketch).')  
    parser.add_argument('--tag', dest="tag", action='store', type=str, help='The tag for saving the model', default="chairs_model")
    parser.add_argument('--weight_cls', dest="weight_cls", action='store', type=float, help='The weight_cls hyperparameter', default="1.0")
    args = parser.parse_args()
    opt_ = SketchOptions(tag=args.tag, weight_cls = args.weight_cls)
    sfs = TrainerCombined(opt_)
    sfs.train()
