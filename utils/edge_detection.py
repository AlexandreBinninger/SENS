import numpy as np

from custom_types import *
from utils import files_utils, train_utils, mesh_utils, image_utils
import cv2 as cv
from PIL import  Image, ImageDraw
# from shapely.geometry import Polygon
# from shapely.geometry import LineString, MultiPolygon
# from shapely.ops import polygonize, unary_union
# import imageio

def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2d = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2d = gaussian_2d / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2d = gaussian_2d / np.sum(gaussian_2d)
    return gaussian_2d


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2d_numerator = x
    sobel_2d_denominator = (x ** 2 + y ** 2)
    sobel_2d_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2d = sobel_2d_numerator / sobel_2d_denominator
    return sobel_2d


def get_thin_kernels(start=0, end=360, step=45):
    k_thin = 3  # actual size of the directional kernel
    # increase for a while to avoid interpolation when rotating
    k_increased = k_thin + 2

    # get 0° angle directional kernel
    thin_kernel_0 = np.zeros((k_increased, k_increased))
    thin_kernel_0[k_increased // 2, k_increased // 2] = 1
    thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

    # rotate the 0° angle directional kernel to get the other ones
    thin_kernels = []
    for angle in range(start, end, step):
        (h, w) = thin_kernel_0.shape
        # get the center to not rotate around the (0, 0) coord point
        center = (w // 2, h // 2)
        # apply rotation
        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
        kernel_angle_increased = cv.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv.INTER_NEAREST)

        # get the k=3 kerne
        kernel_angle = kernel_angle_increased[1:-1, 1:-1]
        is_diag = (abs(kernel_angle) == 1)  # because of the interpolation
        kernel_angle = kernel_angle * is_diag  # because of the interpolation
        thin_kernels.append(kernel_angle)
    return thin_kernels


class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 use_cuda=False):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'

        # gaussian

        gaussian_2d = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=(k_gaussian, k_gaussian),
                                         padding=(k_gaussian // 2, k_gaussian // 2),
                                         bias=False)
        self.gaussian_filter.weight.data = torch.from_numpy(gaussian_2d).float().unsqueeze(0).unsqueeze(0)

        # sobel

        sobel_2d = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=(k_sobel, k_sobel),
                                        padding=(k_sobel // 2, k_sobel // 2),
                                        bias=False)
        self.sobel_filter_x.weight.data = torch.from_numpy(sobel_2d).float().unsqueeze(0).unsqueeze(0)
        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=(k_sobel, k_sobel),
                                        padding=(k_sobel // 2, k_sobel // 2),
                                        bias=False)
        self.sobel_filter_y.weight.data = torch.from_numpy(sobel_2d.T).float().unsqueeze(0).unsqueeze(0)
        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)
        self.directional_filter.weight.data = torch.from_numpy(directional_kernels).float().unsqueeze(1)

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=(3, 3),
                                    padding=(1, 1),
                                    bias=False)
        self.hysteresis.weight.data = torch.from_numpy(hysteresis).float().unsqueeze(0).unsqueeze(0)

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        # gaussian

        for c in range(C):
            blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c+1])

        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180 # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges

        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1

        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges



def contour_length(contour: ARRAY) -> float:
    contour = torch.from_numpy(contour).float()
    delta = (contour[1:] - contour[:-1]).norm(2, -1).sum()
    return delta


def contours2image(contours: ARRAYS, bg: Tuple[int, int, int] = (255, 255, 255), res: int = 256, thickness=1):
    image = np.zeros((res, res, 3), dtype=np.uint8)
    image[:, :] = bg
    if DEBUG:
        lengths = [contour_length(contour) for contour in contours]
    contours = [contour for contour in contours if contour_length(contour) > 100 / (256 / res) ** 2]
    image = cv.drawContours(image, contours, -1, (0, 0, 0), thickness)
    # image = Image.new("RGB", (res, res), color=bg)
    # draw = ImageDraw.Draw(image)
    # for countour in contours:
    #     draw.line(countour, fill='black', width=5)
    return image


def image2points(bg: T):
    silhouette = bg.numpy()
    contours, hierarchy = cv.findContours(silhouette.astype(np.uint8), cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_KCOS)
    return contours
    # contours = torch.from_numpy(contours[0][:, 0]).float()
    # contours_ = torch.cat((contours, contours[0].unsqueeze(0)), dim=0)
    # lengths = (contours_[1:] - contours_[:-1]).norm(2, 1)
    # min_length = lengths.min() / 2
    # max_length = lengths.max()
    # while max_length > min_length:
    #     contours_ = torch.cat((contours, contours[0].unsqueeze(0)), dim=0)
    #     lengths = (contours_[1:] - contours_[:-1]).norm(2, 1)
    #     mask = lengths.gt(min_length)
    #     max_length = lengths.max()
    #     mids = (contours_[1:] + contours_[:-1]) / 2
    #     mask = torch.stack((torch.ones(contours.shape[0], dtype=torch.bool), mask), dim=1).flatten()
    #     contours = torch.stack((contours, mids), dim=1).view(-1, 2)[mask]
    # vs = contours
    # vs = mesh_utils.to_unit_sphere(vs, scale=10)
    # return vs