import cv2 as cv
import os
import os.path



# for frame in frames
# load matching ones in images
# do multiplication
for dir in ['trainA']:
    image_root = f"../frames/train/{dir}/"
    mask_root = f"../frames/mask/{dir}/"
    patch_root= f"../patches/mask/{dir}/"

    images = sorted(os.listdir(image_root))
    patches = sorted(os.listdir(patch_root))
    for i, file in enumerate(images):
        print(f'{i+1} out of {len(images)}')
        if os.path.isfile(mask_root+file.replace('.jpg', '.png')):
            img = cv.imread(image_root+file)
            # mask = cv.imread(mask_root+file.replace('.jpg', '.png'))
            
            counter = 0
            while file.replace('.jpg', '')+f'_{counter}.png' in patches:
                patch = cv.imread(patch_root+file.replace('.jpg', '')+f'_{counter}.png')

                extracted_patch = cv.bitwise_and(img, patch)
                cv.imwrite(f"../patches/extracted/{dir}/{file.replace('.jpg', '')}_{counter}.png", extracted_patch)

                counter += 1

            # extracted = cv.bitwise_and(img, mask)

            # cv.imwrite(f'../frames/extracted/{dir}/{file}', extracted)
