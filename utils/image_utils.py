import numpy as np

import constants
import scipy.ndimage
from matplotlib import pyplot as plt
from PIL import Image
from custom_types import *
from custom_types import T, ARRAY
from utils import files_utils
import imageio


def is_grayscale(image: ARRAY) -> bool:
    if image.ndim == 2 or image.shape[-1] == 1:
        return True
    mask = np.equal(image[:, :, 0], image[:, :, 1]) * np.equal(image[:, :, 2], image[:, :, 1])
    return mask.all()

def crop_square(image:ARRAY) -> ARRAY:
    h, w, c = image.shape
    offset = abs(h - w)
    if h > w:
        image = image[offset // 2:  h - offset + offset // 2]
    elif h < w:
        image = image[:, offset // 2:  w - offset + offset // 2]
    return image


def resize(image_arr: ARRAY, max_edge_length: int) -> ARRAY:
    h, w, c = image_arr.shape
    max_edge = max(w, h)
    if max_edge < max_edge_length:
        return image_arr
    if c == 1:
        image_arr = image_arr[:, :, 0]
    image = Image.fromarray(image_arr)
    s = max_edge_length / float(max_edge)
    size = (int(w * s), int(h * s))
    image = V(image.resize(size, resample=Image.BICUBIC))
    if c == 1:
        image = np.expand_dims(image, 2)
    return image


def rba_to_rgb(rgba_image):
    rgba_image.load()
    background = Image.new("RGB", rgba_image.size, (255, 255, 255))
    background.paste(rgba_image, mask=rgba_image.split()[3])
    return V(background)


def rba_to_rgba(rgba_image):
    rgba_image.load()
    background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
    background.paste(rgba_image, mask=rgba_image.split()[3])
    return V(background)


def rba_to_rgb_arr(image):
    rgba_image = (image * 255).astype(np.uint8)
    rgba_image = Image.fromarray(rgba_image)
    return rba_to_rgb(rgba_image)


def rba_to_rgba_path(path: str):
    rgba_image = Image.open(path)
    return rba_to_rgba(rgba_image)


def rba_to_rgb_path(path: str):
    rgba_image = Image.open(path)
    return rba_to_rgb(rgba_image)


def gif_group(group, folder, interval, name, mp4=False, loop = 0, reverse: bool = False):
    if reverse:
        group_ = group.copy()
        group_.reverse()
        group = group + group_
        interval_ = interval
    else:
        if type(interval) is not list:
            interval_ = [interval] * len(group)
            interval_[0] = 1
        else:
            interval_ = interval
    extension = 'mp4' if mp4 else 'gif'
    if mp4:
        if interval < 1.:
            fps = (1. / interval)
        else:
            fps = interval
        imageio.mimsave(f'{folder}{name}.{extension}', group, fps=fps)
    else:
        imageio.mimsave(f'{folder}{name}.{extension}',
                        group, duration=interval_, loop=loop)

def sort_again(paths: List[List[str]]) ->  List[List[str]]:
    out = []
    long_paths = []
    for path in paths:
        name = path[1]
        split_name = name.split('_')
        if len(split_name[1]) == 4:
            long_paths.append(path)
        else:
            if len(long_paths) > 0:
                out = out[:-1] + long_paths + [out[-1], path]
                long_paths = []
            else:
                out.append(path)
    if len(long_paths) > 0:
        out = out[:-1] + long_paths + [out[-1]]
    return out

def compare(x, y):
    names_a = x[1].split('_')
    names_b = y[1].split('_')
    if int(names_a[1]) < int(names_b[1]):
        return -1
    elif int(names_a[1]) > int(names_b[1]):
        return 1
    elif len(names_a) == 3 and len(names_b) == 3:
        if int(names_a[2]) < int(names_b[2]):
            return -1
        else:
            return 1
    elif len(names_a) == 3:
        if len(names_b[1]) == len(names_a[1]):
            return 1
        return -1
    elif len(names_b) == 3:
        if len(names_b[1]) == len(names_a[1]):
            return -1
        return 1
    elif len(names_a[1]) == 4:
        return -1
    else:
        return 1


def gifed(folder: str, interval: float, name: str, filter_by: Optional[Callable[[List[str]], bool]] = None,
          loop: int = 0, split: int = 1, reverse: bool = True, mp4=False, is_alpha: bool = False):
    folder = files_utils.add_suffix(folder, "/")
    files = files_utils.collect(folder, '.png')
    if filter_by is not None:
        files = list(filter(filter_by, files))
    interval = len(files) * [interval]
    interval[-4] = .3
    interval[-1] = .3
    print(sum(interval))
    if len(files) > 0:
        if is_alpha:
            images = [[rba_to_rgb_path(''.join(file)) for file in files]]
        else:
            images = [[imageio.imread(''.join(file)) for file in files]]
        if split > 1:
            images_ = []
            for i, image in enumerate(images[0]):
                if i % split == 0:
                    images_.append([])
                images_[-1].append(image)
            images = images_
        for i, group in enumerate(images):
            name_ = f'{name}{str(i) if split > 1 else ""}'
            gif_group(group, folder, interval, name_, mp4, loop, reverse)


def get_offsets(image_, margin):
    white = np.equal(image_.sum(2), 255 * 3)
    white_rows = np.equal(white.sum(1), image_.shape[1])
    white_cols = np.equal(white.sum(0), image_.shape[0])
    offset_top, offset_bottom = np.where(~white_rows)[0].min() - margin, np.where(~white_rows)[0].max() + 1 + margin
    offset_left, offset_right = np.where(~white_cols)[0].min() - margin, np.where(~white_cols)[0].max() + 1 + margin
    offset_top, offset_left = max(offset_top, 0), max(offset_left, 0)
    offset_bottom, offset_right = min(offset_bottom, image_.shape[0]), min(offset_right, image_.shape[1])
    return offset_top, offset_left, offset_bottom, offset_right


def crop_white(root: str, in_place=True, offset=1, as_first=False, alpha=True):
    paths = files_utils.collect(root, '.jpg', '.png')
    offset_top = offset_left = 1000000
    offset_bottom = offset_right = -23
    if as_first:
        for path in paths:
            image = files_utils.load_image(''.join(path))
            offset_top_, offset_left_, offset_bottom_, offset_right_ = get_offsets(image, offset)
            offset_top, offset_left = min(offset_top, offset_top_), min(offset_left, offset_left_)
            offset_bottom, offset_right = max(offset_bottom, offset_bottom_), max(offset_right, offset_right_)
    for path in paths:
        image = files_utils.load_image(''.join(path))
        if not in_place:
            new_path = list(path)
            new_path[1] = f'{new_path[1]}_cropped'
        else:
            new_path = path
        white = np.equal(image.sum(2), 255 * 3)
        if not as_first:
            offset_top, offset_left, offset_bottom, offset_right = get_offsets(image, offset)
        image = image[offset_top: offset_bottom, offset_left: offset_right]
        if alpha:
            alpha_ = (1 - white[offset_top: offset_bottom, offset_left: offset_right].astype(image.dtype)) * 255
            alpha_ = np.expand_dims(alpha_, 2)
            image = np.concatenate((image, alpha_), axis=2)
        files_utils.save_image(image, ''.join(new_path))


def change_mesh_color(paths, color):
    for path in paths:
        mesh = files_utils.load_mesh(path)
        files_utils.export_mesh(mesh, path, colors=color)


def to_heatmap(vals: Union[T, ARRAY], palette: str = 'coolwarm') -> T:
    shape = vals.shape
    if type(vals) is T:
        vals: ARRAY = vals.detach().cpu().numpy()
    to_reshape = vals.ndim > 1
    if to_reshape:
        vals = vals.flatten()
    vals = (vals * 255).astype(np.uint8)
    colormap = plt.get_cmap(palette)
    np_heatmap = colormap(vals)[:, :3]
    heatmap = torch.from_numpy(np_heatmap).float()
    if to_reshape:
        heatmap = heatmap.view(*shape, 3)
    return heatmap

def make_transition(path_a: str, path_b: str, num_between: int):
    image_a = files_utils.load_image(path_a, 'RGBA').astype(np.float64)
    image_b = files_utils.load_image(path_b, 'RGBA').astype(np.float64)
    diff = image_b - image_a
    alpha = np.linspace(0, 1, num_between + 2)[1:-1]
    for i in range(len(alpha)):
        image = image_a + alpha[i] * diff
        image = image.astype(np.uint8)
        files_utils.save_image(image, f'{path_a}_{i:d}.png')


def get_bounds(*samples):
    bounds = [[np.expand_dims(func(sample, 0), 0) for sample in samples] for func in [np.min, np.max]]
    bounds = [np.expand_dims(func(np.concatenate(bound), 0),0) for bound, func in zip(bounds,[np.min, np.max])]
    return np.concatenate(bounds, 0)


def blend_images(images: List[V], blend_height: int, blend_width: int, rows: int) -> List[V]:
    cols = len(images) // rows
    for i in range(cols - 1):
        for j in range(rows):
            image_index = i + j * cols
            blend_a = images[image_index][:, -blend_width:]
            blend_b = images[image_index + 1][:, : blend_width]
            if blend_b.shape[-1] == 4:
                ma = blend_b[:, :, -1] > blend_a[:, :, -1]
            else:
                ma = blend_b > blend_a
            blend_a[ma] = blend_b[ma]
            images[image_index][:, -blend_width:] = blend_a
            images[image_index + 1] = images[image_index + 1][:, blend_width:]
    for i in range(rows - 1):
        for j in range(cols):
            image_index = i * cols + j
            blend_a = images[image_index][-blend_height:, :]
            blend_b = images[image_index + cols][: blend_height, :]
            if blend_b.shape[-1] == 4:
                ma = blend_b[:, :, -1] > blend_a[:, :, -1]
            else:
                ma = blend_b > blend_a
            blend_a[ma] = blend_b[ma]
            images[image_index][-blend_height:, :] = blend_a
            images[image_index + cols] = images[image_index + cols][blend_height:, :]
    return images


def make_pretty(np_images: List[ARRAY], offset=.01, blend=0.35, rows=1) -> Image.Image:
    if type(offset) is not tuple:
        offset = [offset] * 4
    if type(blend) is not tuple:
        blend = [blend] * 2
    offset = [- np_images[0].shape[idx % 2] if off == 0 else int(np_images[0].shape[idx % 2] * off) for idx, off in enumerate(offset)]
    cols = len(np_images) // rows
    np_images = np_images[: cols * rows]

    # offset_height, offset_width = int(np_images[0].shape[0] * offset ), int(np_images[0].shape[1] * offset)
    blend_height, blend_width = int(np_images[0].shape[0] * blend[0]), int(np_images[0].shape[1] * blend[1])
    np_images = [image[offset[3]: - offset[1], offset[0]: - offset[2]] for image in np_images]
    # np_images = [image[offset_height: - offset_height, offset_width: - offset_width] for image in np_images]
    if blend[0] != 0:
        np_images = blend_images(np_images, blend_height, blend_width, rows)
    np_images = [np.concatenate(np_images[i * cols: (i + 1) * cols], axis=1) for i in range(rows)]
    im = np.concatenate(np_images, axis=0)
    im = Image.fromarray(im)
    return im


def images_to_numpy(*paths: List[str]) -> ARRAYS:
    images = [V(Image.open(''.join(path))) for path in paths]
    return images
