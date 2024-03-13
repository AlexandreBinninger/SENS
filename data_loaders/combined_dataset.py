
import os
from data_loaders import clipasso_dataset, sketch_dataset, prosketch_dataset
from custom_types import *
from utils import files_utils, edge_detection, rotation_utils
import options
import constants
import torchvision
import matplotlib.pyplot as plt
import enum
import random
from utils.train_utils import Logger

def get_random_numpy(n : int, max:int):
    if n <= max:
        return np.random.choice(max, size=n, replace=False)
    else:
        return np.arange(max)

class CombinedDs(Dataset):
    def __getitem__(self, item: int):
        if item < self.len_sketch_ds :
            image_full, image_masked, masks, zh = self.sketch_ds[item]
            weight_masked = torch.ones((1, 1))
            return weight_masked, image_full, image_masked, masks, zh
        elif (item < self.len_sketch_ds + self.len_clipasso_ds):
            idx = item - self.len_sketch_ds
            image_full, zh = self.clipasso_ds[idx]
            image_masked = image_full
            masks = torch.ones((16, 1))
            weight_masked = torch.zeros((1, 1))
            return weight_masked, image_full, image_masked, masks, zh
        else:
            idx = item - self.len_sketch_ds - self.len_clipasso_ds
            image_full, zh = self.prosketch_ds[idx]
            image_masked = image_full
            masks = torch.ones((16, 1))
            weight_masked = torch.zeros((1, 1))
            return weight_masked, image_full, image_masked, masks, zh

    def __len__(self):
        return self.len_sketch_ds + self.len_clipasso_ds + self.len_prosketch_ds
    

    def set_augmentation(self, augment : bool):
        self.sketch_ds.set_augmentation(augment)
        self.clipasso_ds.set_augmentation(augment)
        if self.prosketch_ds is not None:
            self.prosketch_ds.set_augmentation(augment)

    def __init__(self, opt_skds: options.SketchOptions, opt_clds: options.SketchOptions, opt_prosk: Optional[options.SketchOptions]):
        self.sketch_ds = sketch_dataset.SketchDs(opt_skds)
        self.len_sketch_ds = len(self.sketch_ds)
        self.clipasso_ds = clipasso_dataset.ClipassoDs(opt_clds)
        self.len_clipasso_ds = len(self.clipasso_ds)
        if opt_skds.name_cat== "chairs" and opt_prosk is not None:
            self.prosketch_ds = prosketch_dataset.ProsketchDs(opt_prosk)
            self.len_prosketch_ds = len(self.prosketch_ds)
        else:
            self.prosketch_ds = None
            self.len_prosketch_ds = 0
            
def main(): 
    from utils import train_utils
    ds = CombinedDs(options.SketchOptions(spaghetti_tag='chairs_sym_hard'), options.SketchOptions(spaghetti_tag='chairs_sym_hard'), options.SketchOptions(data_tag="prosketch/original/"))
    for _ in range(0, 100):
        item = random.randint(0, len(ds))
        weight_masks, sketches_full, sketches, _, _ = ds[item]
        if weight_masks.sum().item() > 0:
            plt.imshow(sketches_full[0], cmap='gray')
            plt.show()
        plt.imshow(sketches[0], cmap='gray')
        plt.show()
    return 0

if __name__ == '__main__':
    main()
