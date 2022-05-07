import cv2 as cv
import os


# for frame in frames
# load matching ones in images
# do multiplication
for dir in ['trainA', 'trainB']:
    image_root = f"../frames/train/{dir}/"
    mask_root = f"../frames/mask/{dir}/"
    patch_root= f"../patches/mask/{dir}/"

    counter = 0
    images = os.listdir(image_root)
    patches = os.listdir(patch_root)
    for i, file in enumerate(images):
        print(f'{i} out of {len(images)}')
        img = cv.imread(image_root+file)
        mask = cv.imread(mask_root+file.replace('.jpg', '.png'))

        while counter < len(patches) and file.replace('.jpg', '') == '_'.join(patches[counter].split('_')[:-1]):
            patch = cv.imread(patch_root+patches[counter])

            extracted_patch = cv.bitwise_and(img, patch)
            cv.imwrite(f'../patches/extracted/{dir}/{patches[counter]}', extracted_patch)

            counter += 1


        extracted = cv.bitwise_and(img, mask)

        cv.imwrite(f'../frames/extracted/{dir}/{file}', extracted)