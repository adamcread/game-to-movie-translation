import glob
import os
import cv2 as cv
import json 

coco_dataset = {
    "info": {},
    "licenses": [
        {}
    ],
    "images": [],
    "annotations": [],
    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person"
        },
    ]
}

os.chdir('../frames/')
files = glob.glob('train/train*/*', recursive=True)[:10]
for id, file in enumerate(files):
    print(f'{id} out of {len(files)}')
    image = cv.imread(file)

    coco_dataset["images"].append({
        "license": 0,
        "file_name": file,
        "width": image.shape[1],
        "height": image.shape[0],
        "id": id
    })

    coco_dataset["annotations"].append({
        "segmentation": [],
        "iscrowd": 1,
        "image_id": id,
        "id": id,
        "category_id": 1,
        "bbox": [],
        "area": 0
    })

with open('../annotation/test_temp.json', 'w+') as fp:
    json.dump(coco_dataset, fp, indent=4)
