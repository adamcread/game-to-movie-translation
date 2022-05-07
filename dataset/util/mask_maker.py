from pycocotools.mask import decode
from pycocotools.coco import COCO
import numpy as np
import numpy as np
import cv2 as cv
import json


def NMS(boxes, overlapThresh = 0.7):
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return []

    boxes = boxes[boxes[:, 4].argsort()]

    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        # Create temporary indices  
        temp_indices = indices[indices != i]

        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])

        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap

        overlap = (w * h) / areas[temp_indices]

        # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]

    return boxes[indices]

if __name__ == '__main__':
    segmentation_json = json.loads(open('../annotation/test.segm.json', 'r').read())

    id = 0
    bboxes = {}
    for ann_id, segmentation in enumerate(segmentation_json):
        id = int(segmentation['image_id'])
        bboxes[id] = bboxes.get(id, [])
        score = float(segmentation["score"])

        if score > 0.90:
            bboxes[id].append([*segmentation['bbox'], score, segmentation["segmentation"]])

    dataset = COCO(annotation_file="../annotation/test.json")
    for id in bboxes.keys():
        image_bboxes = np.array([*bboxes[id]])
        suppressed_bboxes = np.array(NMS(image_bboxes))


        image = dataset.loadImgs(ids=id)[0]
        file_name = image["file_name"]

        height = image["height"]
        width = image["width"]
        mask = np.zeros((height, width))

        if len(suppressed_bboxes):
            for i, annotation in enumerate(suppressed_bboxes):
                patch = decode(annotation[5])
                cv.imwrite(f'../patches/extracted/{file_name}_{i}.png', patch*255)

                mask = mask + patch
    
        print(file_name)
        cv.imwrite(f'../frames/mask/{file_name}.png', mask*255)

