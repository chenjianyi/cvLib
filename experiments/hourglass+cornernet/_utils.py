import random
import numpy as np
import cv2

def normalize(image, mean, std):
    image -= mean
    image /= std

def crop_image(image, center, size):
    cty, ctx = center
    height, width = size
    img_height, img_width = image.shape[0: 2]
    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)
    
    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, img_width)
    y0, y1 = max(0, cty - height // 2), min(ctx + height // 2, img_height)

    (left, right) = (ctx - x0, x1 - ctx)
    (top, bottom) = (cty - y0, y1 - cty)

    cropped_cty, cropped_ctx = height // 2, width // 2
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = image[y0: y1, x0: x1, :]

    border = np.array([
        cropped_cty - top,
        cropped_cty + bottom,
        cropped_ctx - left,
        cropped_ctx + right 
    ], dtype=np.float32)

    offset = np.array([cty - height // 2, ctx - width // 2])

    return cropped_image, border, offset

def full_image_crop(image, detections):
    detections = detections.copy()
    height, width, channel = image.shape

    max_hw = max(height, width)
    center = [height // 2, width // 2]
    size = [max_hw, max_hw]

    image, border, offset = crop_image(image, center, size)
    detections[:, 0:4:2] += border[2]
    detections[:, 1:4:2] += border[0]
    return image, detections

def _get_border(border, size):
    i = 1
    while ( (size - border // i) <= border // i ):
        i *= 2
    return border // i

def random_crop(image, detections, random_scales, view_size, border=64):
    view_height, view_width = view_size
    image_height, image_width = image.shape[0: 2]

    scale = np.random.choice(random_scales)
    height = int(view_height * scale)
    width = int(view_width * scale)

    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

    w_border = _get_border(border, image_width)
    h_border = _get_border(border, image_height)

    ctx = np.random.randint(low=w_border, high=image_width - w_border)
    cty = np.random.randint(low=h_border, high=image_height - h_border)

    x0, x1 = max(ctx - width // 2, 0), min(ctx + width // 2, image_width)
    y0, y1 = max(cty - height // 2, 0), min(cty + height // 2, image_height)

    left_w, right_w = ctx - x0, x1 - ctx
    top_h, bottom_h = cty - y0, y1 - cty  

    cropped_ctx, cropped_cty = width // 2, height // 2
    x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)
    y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)
    cropped_image[y_slice, x_slice, :] = image[y0: y1, x0: x1, :]

    cropped_detections = detections.copy()
    cropped_detections[:, 0:4:2] -= x0
    cropped_detections[:, 1:4:2] -= y0
    cropped_detections[:, 0:4:2] += cropped_ctx - left_w
    cropped_detections[:, 1:4:2] += cropped_cty - top_h

    return cropped_image, cropped_detections

def resize_image(image, detections, size):
    detections = detections.copy()
    height, width = image.shape[0: 2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))

    height_ratio = new_height / height
    width_ratio = new_width / width
    detections[:, 0: 4: 2] *= width_ratio
    detections[:, 1: 4: 2] *= height_ratio
    return image, detections

def clip_detections(image, detections):
    detections = detections.copy()
    height, width = image.shape[0: 2]
    detections[:, 0: 4: 2] = np.clip(detections[:, 0: 4: 2], 0, width - 1)
    detections[:, 1: 4: 2] = np.clip(detections[:, 1: 4: 2], 0, height - 1)
    keep_inds = ((detections[:, 2] - detections[:, 0]) > 0) & \
                ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    return detections

def _brightness(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def _blend(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def _contrast(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    _blend(alpha, image, gs_mean)

def _saturation(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    _blend(alpha, image, gs[:, :, None])

def color_jittering(data_rng, image, var=0.4):
    functions = [_brightness, _contrast, _saturation]
    random.shuffle(functions)
    gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, var)

def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2  - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)

def _gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m: m+1,-n: n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1, delte=6):
    diameter = 2 * radius + 1
    gaussian = _gaussian2D((diameter, diameter), sigma=diameter / delte)

    x, y = center

    height, width = heatmap.shape[0: 2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def create_attention_mask(atts, ratios, sizes, detections):
    for det in detections:
        width = det[2] - det[0]
        height = det[3] - det[1]
        max_hw = max(width, height)
        for att, ratio, size in zip(attrs, ratios, sizes):
            if max_hw >= size[0] and max_hw <= size[1]:
                x = (det[0] + det[2]) / 2
                y = (det[1] + det[3]) / 2
                x = (x / ratio).astype(np.int32)
                y = (y / ratio).astype(np.int32)
                att[y, x] = 1

def rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0: 4: 2], detections[..., 1: 4: 2]  # bs * num_dets * 2
    xs /= ratios[:, 1][:, None, None]  # bs * num_dets * 2
    ys /= ratios[:, 0][:, None, None]  # bs * num_dets * 2 
    xs -= borders[:, 2][:, None, None]  # bs * num_dets * 2 
    ys -= borders[:, 0][:, None, None]  # bs * num_dets * 2 
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)
