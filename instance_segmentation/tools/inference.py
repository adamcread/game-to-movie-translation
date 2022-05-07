from mmcv import Config, mkdir_or_exist
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os

PREFIX = '../dataset/frames/test_segmentation/'
cfg = Config.fromfile('./configs/custom/detectors_instaboost.py')

model = init_detector(cfg, checkpoint='./checkpoints/epoch_2.pth', device='cpu')
model.CLASSES = cfg.classes

for image in os.listdir(PREFIX):
    img= PREFIX + image
    print(image)
    result= inference_detector(model, img)
    model.show_result(
        img, 
        result, out_file='../dataset/patches/overall/'+image,
    )
# score threshold