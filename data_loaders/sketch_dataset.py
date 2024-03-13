from PIL import Image
from custom_types import *
from utils import files_utils, edge_detection, rotation_utils
import options
import constants
import torchvision
from data_loaders import augment_clipcenter
import os
import multiprocessing as mp

autumn = [[1.0000, 0.1020, 0.0000, 1.0000],
        [1.0000, 0.1569, 0.0000, 1.0000],
        [1.0000, 0.2078, 0.0000, 1.0000],
        [1.0000, 0.2627, 0.0000, 1.0000],
        [1.0000, 0.3137, 0.0000, 1.0000],
        [1.0000, 0.3686, 0.0000, 1.0000],
        [1.0000, 0.4196, 0.0000, 1.0000],
        [1.0000, 0.4745, 0.0000, 1.0000],
        [1.0000, 0.5255, 0.0000, 1.0000],
        [1.0000, 0.5804, 0.0000, 1.0000],
        [1.0000, 0.6314, 0.0000, 1.0000],
        [1.0000, 0.6863, 0.0000, 1.0000],
        [1.0000, 0.7373, 0.0000, 1.0000],
        [1.0000, 0.7922, 0.0000, 1.0000],
        [1.0000, 0.8431, 0.0000, 1.0000],
        [1.0000, 0.8980, 0.0000, 1.0000]]

class SketchDs(Dataset):

    @staticmethod
    def split_by_range(item: T, range_: T) -> TS:
        out = []
        start = 0
        for i in range(len(range_)):
            out.append(item[start: start + range_[i]])
            start += range_[i]
        return out

    def create_contour_image(self, flat_bg: T, th: Optional[int] = None):
        inside = flat_bg.view(self.res, self.res).ne(0.)
        if th is None:
            th = 1 + torch.randint(3, (1,)).item()
        contours = edge_detection.image2points(inside)
        image = edge_detection.contours2image(contours, res=self.res, thickness=th)
        return image, th

    def get_zh(self, item: int) -> T:
        return self.zh[item]

    def select_parts(self, item: int) -> T:
        vec = self.mus[item]
        mu_z = vec[:, torch.randint(3, (1, )).item()]
        random_down_top_order = mu_z.argsort(dim=-1).argsort(-1)

        num_parts = (self.opt.min_split + torch.randint(self.opt.max_split - self.opt.min_split, size=(1,))).item()
        start_index = torch.randint(vec.shape[0] - num_parts, (1,)).item()
        mask = random_down_top_order.le(num_parts + start_index) * random_down_top_order.ge(start_index)

        return mask


    @staticmethod
    def merge_layers(layers: TS, masks: TS, supports: TS, part_select: T) -> T:
        
        try:
            if layers[0].dim() == 1:
                bg = torch.zeros(masks[0].shape[0], dtype=layers[0].dtype)
            else:
                bg = torch.zeros(masks[0].shape[0], layers[0].shape[1], dtype=layers[0].dtype)
            previous_not_selected = torch.ones(supports[0].shape[0], dtype=torch.bool)
            relevant_pixels = masks[0].clone()
            for j in range(len(supports)):
                select_ = torch.zeros(supports[j].shape[0], dtype=torch.bool)
                for k in range(part_select.shape[0]):
                    if part_select[k]:
                        select_[supports[j][:, k]] = True
                select_ = select_ * previous_not_selected
                if select_.any():
                    select_mask = relevant_pixels.clone()
                    select_mask[relevant_pixels] = select_
                    bg[select_mask] = layers[j][select_ * previous_not_selected]
                if j < len(supports) - 1:
                    relevant_pixels[relevant_pixels.clone()] = masks[j + 1]
                    previous_not_selected = ~select_[masks[j + 1]] * previous_not_selected[masks[j + 1]]
            return bg
        except Exception as e:
            print(e)
            return bg

    def get_full_render(self, mask, points, light_dir: Optional[T] = None, color: Optional[T] = None):
        if mask is None:
            mask = points.sum(-1).ne(0)
            tiled_mask = points
        else:
            tiled_mask = torch.zeros(256 ** 2)
            tiled_mask[mask.bool()] = points.float()
        if light_dir is None:
            light_dir = nnf.normalize(torch.tensor([-1., 1., -.8]), 2, 0)
        if color is None:
            color = torch.tensor([[[1., 1., 1.]]])
        mask = mask.float()
        tiled_mask = tiled_mask.permute(1, 0).view(1, 3, 256, 256)
        tiled_mask_padded = nnf.pad(tiled_mask, (1, 1, 1, 1))
        dy = (tiled_mask_padded[:, :, :-2, 1:-1] - tiled_mask_padded[:, :, 2:, 1:-1])
        dx = (tiled_mask_padded[:, :, 1:-1, :-2] - tiled_mask_padded[:, :, 1:-1, 2:])
        cross = torch.cross(dy, dx, dim=1)
        cross = nnf.normalize(cross, 2, dim=1).permute(0, 2, 3, 1)[0]
        pixels = torch.einsum('hwc,c->hw', cross, light_dir)
        pixels = (pixels + 1) / 2
        pixels = (0.3 + 0.7 * pixels.unsqueeze(2)) * color
        pixels = pixels * mask.view(256, 256, 1) + (1 - mask.view(256, 256, 1))
        return pixels

    def get_contour(self, depth_maps, masks, supports, part_select, th: Optional[int] = None) -> T:
        image_full = self.get_full_contour(masks[0], depth_maps[0])
        bg = self.merge_layers(depth_maps, masks, supports, part_select)
        image_masked = self.get_full_contour(None, bg, th)
        return self.augment_masked(image_masked, image_full)

    def get_render(self, points, masks, supports, part_select) -> T:
        bg = self.merge_layers(points, masks, supports, part_select)
        return self.get_full_render(None, bg.float())

    def get_layered_image(self, item):
        points, depth_maps, masks, supports = self.load_item(item)
        images = []
        for i in range(16):
            part_select = torch.zeros(16).bool()
            part_select[i] = True
            bg = self.merge_layers(depth_maps, masks, supports, part_select)
            bg = bg.view(256, 256).ne(0)
            image = torch.zeros(256, 256, 4)
            image[bg] = torch.tensor(autumn[i])
            image = (image * 255).numpy().astype(np.uint8)
            images.append(image)
        return images

    @staticmethod
    def get_creases(mask, points):
        mask = mask.float()
        tiled_mask = torch.zeros(256 ** 2, 3)
        tiled_mask[mask.bool()] = points.float()
        tiled_mask = tiled_mask.permute(1, 0).view(1, 3, 256, 256)
        tiled_mask_padded = nnf.pad(tiled_mask, (1, 1, 1, 1))
        dy = (tiled_mask_padded[:, :, :-2, 1:-1] - tiled_mask_padded[:, :, 2:, 1:-1])
        dx = (tiled_mask_padded[:, :, 1:-1, :-2] - tiled_mask_padded[:, :, 1:-1, 2:])
        cross = torch.cross(dy, dx, dim=1)
        cross = nnf.normalize(cross, 2, dim=1)
        tiled_cross = nnf.pad(cross, (1, 1, 1, 1))
        dy = torch.einsum('bchw,bchw->bhw', tiled_cross[:, :, :-2, 1:-1], tiled_cross[:, :, 2:, 1:-1]).abs()
        dx = torch.einsum('bchw,bchw->bhw', tiled_cross[:, :, 1:-1, :-2], tiled_cross[:, :, 1:-1, 2:]).abs()
        d_max = torch.cat((dy, dx), dim=0).min(0)[0]
        d_mask = d_max < .2
        d_mask[~mask.bool().view(256, 256)] = 0
        return d_mask

    @staticmethod
    def remove_by_other(mask, image, with_pad: bool, th: float):
        if with_pad:
            image = nnf.max_pool2d(image, 3, 1, 1)
        image = image.mean(1).lt(th)[0]
        mask = mask * image
        return mask

    def get_full_contour(self, mask, depth_map, th: Optional[int] = None):
        if mask is None:
            mask = depth_map.ne(0)
            tiled_mask = depth_map.float()
        else:
            tiled_mask = torch.zeros(256 ** 2)
            tiled_mask[mask.bool()] = depth_map.float()
        tiled_mask = tiled_mask.view(1, 1, 256, 256)
        mask = mask.float()
        tr = .2 + torch.rand(1) * .1
        tiled_mask_padded = nnf.pad(tiled_mask, (1, 1, 1, 1))
        dy = (tiled_mask_padded[:, :, 1:-1, 1:-1] - tiled_mask_padded[:, :, 2:, 1:-1]).abs()
        dx = (tiled_mask_padded[:, :, 1:-1, 1:-1] - tiled_mask_padded[:, :, 1:-1, 2:]).abs()
        dxb = (tiled_mask_padded[:, :, 1:-1, 1:-1] - tiled_mask_padded[:, :, 1:-1, :-2]).abs()
        dyb = (tiled_mask_padded[:, :, 1:-1, 1:-1] - tiled_mask_padded[:, :, :-2, 1:-1]).abs()
        d_max = torch.cat((dy, dx), dim=1).max(1)[0][0]
        d_max_b = torch.cat((dyb, dxb), dim=1).max(1)[0][0]
        d_max = d_max * mask.view(256, 256)
        d_max_b = d_max_b * mask.view(256, 256)
        d_mask = d_max > tr
        d_mask_b = d_max_b > tr
        d_mask_b = self.remove_by_other(d_mask_b, d_mask.unsqueeze(0).unsqueeze(0).float(), True, 1.)
        d_mask = d_mask + d_mask_b
        contour, th = self.create_contour_image(mask, th)
        d_mask_b = self.remove_by_other(d_mask, torch.from_numpy(255 - contour).float().permute(2, 0, 1).unsqueeze(0), th < 2, 120.)
        contour[d_mask_b] = 0
        return contour

    def split_by_view(self, raw_data, view):
        res = 256
        depth_maps, masks = raw_data["depth_maps"], raw_data["masks"]
        supports, ranges = raw_data["supports"], raw_data["ranges"]
        points = raw_data["points"]
        depth_maps, supports, points = map(lambda x: self.split_by_range(x, ranges[view]),
                                           (depth_maps[view], supports[view], points[view]))
        mask_range = ranges[view].roll(1)
        mask_range[0] = res ** 2
        masks = self.split_by_range(masks[view], mask_range)
        return points, depth_maps, masks, supports

    def load_item(self, item: int, select: Optional[int] = None) -> Tuple[TS, ...]:
        res = 256
        raw_data = files_utils.load_pickle(''.join(self.paths[item]))
        supports, ranges = raw_data["supports"], raw_data["ranges"]
        if select is None:
            select = torch.randint(len(supports), (1,)).item()
        return self.split_by_view(raw_data, select)

    def get_renders(self, item: int, view: int):
        points, depth_maps, masks, supports = self.load_item(item, view)
        renders = []
        part_select = torch.ones(self.mus[item].shape[0], dtype=torch.bool)
        renders.append(self.get_render(points, masks, supports, part_select))
        files_utils.imshow(renders[-1])
        return renders

    def get_full_renders(self, item):
        raw_data = files_utils.load_pickle(''.join(self.paths[item]))
        renders = []
        for i in range(len(raw_data["supports"])):
            points, depth_maps, masks, supports = self.split_by_view(raw_data, i)
            part_select = torch.ones(self.mus[item].shape[0], dtype=torch.bool)
            renders.append(self.get_render(points, masks, supports, part_select))
        return renders

    def augment_full(self, image):
        image = Image.fromarray(image)
        if self.augmentation:
            pur = self.get_random()
            image = self.hflip_aug(image)
            if pur < .33:
                image = self.perspective_aug(image)
            elif pur < .66:
                image = self.affine_aug(image)
        image = augment_clipcenter.augment_cropped_square(image)
        image = torch.from_numpy(V(image)).float() / 255.
        image = image.mean(-1).unsqueeze(0)
        return image

    def augment_masked(self, image, image_full):
        image = Image.fromarray(image)
        image_full = Image.fromarray(image_full)
        _, image = augment_clipcenter.augment_cropped_square_fullandcropped(image_full, image_masked=image)
        if self.augmentation:
            pur = self.get_random()
            image = self.hflip_aug(image)
            if pur < .33:
                image = self.perspective_aug(image)
                image_full = self.perspective_aug(image_full)
            elif pur < .66:
                image = self.affine_aug(image)
                image_full = self.affine_aug(image_full)
        image = torch.from_numpy(V(image)).float() / 255.
        image = image.mean(-1).unsqueeze(0)
        return image
    
    def to_torch(self, image):
        return image

    def __getitem__(self, item: int):
        points, depth_maps, masks, supports = self.load_item(item)
        part_select = self.select_parts(item)
        image_full = self.get_full_contour(masks[0], depth_maps[0])
        image_full = self.augment_full(image_full)
        image = self.get_contour(depth_maps, masks, supports, part_select)
        zh = self.get_zh(item)
        return image_full, image, part_select.float().unsqueeze(-1), zh
    
    def getitem_view(self, item: int, view: int):
        points, depth_maps, masks, supports = self.load_item(item, view)
        part_select = self.select_parts(item)
        image_full = self.get_full_contour(masks[0], depth_maps[0])
        image_full = self.augment_full(image_full)
        image = self.get_contour(depth_maps, masks, supports, part_select)
        zh = self.get_zh(item)
        return image_full, image, part_select.float().unsqueeze(-1), zh

    def __len__(self):
        return len(self.paths)
    
    def get_random(self):
        with self.cur_rand.get_lock():  # Acquire the lock before modifying
            self.cur_rand.value += 1
            if self.cur_rand.value >= self.size_random:
                self.random_array = np.random.rand(self.size_random)
                self.cur_rand.value = 0
            return self.random_array[self.cur_rand.value]
    
    def get_random(self):
        with self.cur_rand.get_lock():  # Acquire the lock before modifying
            self.cur_rand.value += 1
            if self.cur_rand.value >= self.size_random:
                self.random_array = np.random.rand(self.size_random)
                self.cur_rand.value = 0
            return self.random_array[self.cur_rand.value]

    def set_augmentation(self, augment : bool):
        self.augmentation = augment

    def __init__(self, opt: options.SketchOptions):
        self.opt = opt
        self.res = 256
        out_root = f'{constants.DATA_ROOT}/dataset_chair_preprocess/{opt.spaghetti_tag}/'
        self.paths = files_utils.collect(out_root, '.pkl')
        self.zh = torch.from_numpy(files_utils.load_np(f'{out_root}zh_{self.opt.z_level}'))
        self.mus = torch.from_numpy(files_utils.load_np(f'{out_root}mu'))
        self.affine_aug = torchvision.transforms.RandomAffine(6, translate=(.078, .078), scale=(.95, 1.05),
                                                                shear=3, fill=(255, 255, 255),
                                                                interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.perspective_aug = torchvision.transforms.RandomPerspective(distortion_scale=0.4, p=1.,fill=(255, 255, 255),
                                                                        interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.hflip_aug = torchvision.transforms.RandomHorizontalFlip(p=.5)
        self.augmentation = True
        self.size_random = 10000
        self.random_array = np.random.rand(self.size_random)
        self.cur_rand = mp.Value('i', -1)