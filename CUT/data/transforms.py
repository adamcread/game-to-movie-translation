
#  if 'crop' in opt.preprocess:
#         if params is None or 'crop_pos' not in params:
#             transform_list.append(transforms.RandomCrop(opt.crop_size))
#         else:
#             transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
from PIL import Image


def __scale_width(img, target_width, crop_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width and oh >= crop_width:
        return img

    w = target_width
    h = int(max(target_width * oh / ow, crop_width))
    return img.resize((w, h), method)