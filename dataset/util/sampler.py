import os
import random
import shutil
import glob 
# root = "../patches/extracted/trainB"
# dest = "../patches/classified"
# images = os.listdir(root)
# print(len(images))
# for image in random.sample(images, k=750):
#     shutil.copyfile(f"{root}/{image}", f"{dest}/{image.replace('.png', '')}_B.png")


files = glob.glob('../patches/classified/val/**/*.png', recursive=True)

# for image in files:
#     shutil.move(image, image.replace('train/', 'val/'))
print(len(files))