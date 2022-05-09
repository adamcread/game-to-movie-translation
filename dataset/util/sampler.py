import os
import random
import shutil

root = "../patches/extracted/trainA"
dest = "../patches/classified"
images = os.listdir(root)
print(len(images))
for image in random.sample(images, k=750):
    shutil.copyfile(f"{root}/{image.split('.')[0]}_A.png")