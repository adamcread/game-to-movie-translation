from unicodedata import category
from pycocotools.coco import COCO
import argparse
import shutil
import os

parser = argparse.ArgumentParser(description='Arguments for coco image filtering.')
parser.add_argument('--json', type=str, help='input json with file annotations')
parser.add_argument('--root', type=str, help='root of files')
parser.add_argument('--dest', type=str, help='destination for files')
parser.add_argument('--categories', type=str, nargs='+', help='space separated list of categories to keep')
parser.add_argument('--file_extension', type=str, help="file extension for targets to be filtered")

args = parser.parse_args()

coco = COCO(annotation_file=args.json)
category_ids = coco.getCatIds(catNms=args.categories)
image_ids = coco.getImgIds(catIds=category_ids)
images = coco.loadImgs(ids=image_ids)

for i, image in enumerate(images):
    print(f'{i} out of {len(image_ids)}')

    file_name = image['file_name'].split('.')[0]+f'.{args.file_extension}'

    if not os.path.exists(args.dest+file_name):
        shutil.copy(args.root+file_name, args.dest+file_name)