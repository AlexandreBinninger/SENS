from PIL import Image
from custom_types import *


def white_column(size_col):
    w_c = np.zeros((size_col, 3), dtype=np.uint8)
    w_c.fill(255)
    return w_c

def white_row(size_row):
    w_c = np.zeros((size_row, 3), dtype=np.uint8)
    w_c.fill(255)
    return w_c

def cropclosest_coordinates(image: ARRAY):
    w, h, d = image.shape
    leftmost, rightmost, uppest, lowest = -1, -1, -1, -1
    w_col = white_column(h)
    w_row = white_row(w)
    count = 0
    while leftmost < 0 and count < h:
        if not (image[:, count, :] == w_row).all():
            leftmost = count
        count +=1
    count = h-1
    while rightmost < 0 and count >= 0:
        if not (image[:, count, :] == w_row).all():
            rightmost = count
        count -=1
    
    count = 0
    while uppest < 0 and count < w:
        if not (image[count, :, :] == w_col).all():
            uppest = count
        count +=1
    count = w-1
    while lowest < 0 and count >= 0:
        if not (image[count, :, :] == w_col).all():
            lowest = count
        count -=1
    leftmost = max(0, leftmost-1)
    rightmost = min(h-1, rightmost+1)
    uppest = max(0, uppest-1)
    lowest = min(w-1, lowest+1)
    return uppest,lowest,leftmost,rightmost

def centeredsquare(image : ARRAY):
    # put the image in the smallest square that it fits in
    w, h, d = image.shape
    side = max(w, h)
    new_image = np.full((side, side, d), 255, dtype=np.uint8)
    if w < h:
        start = (side - w)//2
        end = start+w
        new_image[start:end, :, :] = image
    else:
        start = (side - h)//2
        end = start+h
        new_image[:, start:end, :] = image
    return new_image

def augment_cropped_square(image: Union[ARRAY, Image.Image], res = 256):
    is_numpy = True
    if type(image) is Image.Image:
        image = V(image)
        is_numpy = False
    else:
        if image.dtype != np.uint8:
            raise ValueError # np array should be type uint8
    uppest,lowest,leftmost,rightmost = cropclosest_coordinates(image) 
    image = centeredsquare(image[uppest:lowest,leftmost:rightmost, :])
    image = Image.fromarray(image)
    image = image.resize((res, res), Image.Resampling.BICUBIC)
    if is_numpy:
        image = V(image)
    return image 

def augment_cropped_square_fullandcropped(image_full: Union[ARRAY, Image.Image], image_masked: Union[ARRAY, Image.Image], res = 256):
    is_numpy = True
    if type(image_full) is Image.Image:
        image_full = V(image_full)
        is_numpy = False
    uppest,lowest,leftmost,rightmost = cropclosest_coordinates(image_full) 
    image_full = centeredsquare(image_full[uppest:lowest,leftmost:rightmost, :])
    image_full = Image.fromarray(image_full)
    image_full = image_full.resize((res, res), Image.Resampling.BICUBIC)
    if is_numpy:
        image_full = V(image_full)
    
    is_numpy = True
    if type(image_masked) is Image.Image:
        image_masked = V(image_masked)
        is_numpy = False
    image_masked = centeredsquare(image_masked[uppest:lowest,leftmost:rightmost, :])
    image_masked = Image.fromarray(image_masked)
    image_masked = image_masked.resize((res, res), Image.Resampling.BICUBIC)
    if is_numpy:
        image_masked = V(image_masked)
    return image_full, image_masked

