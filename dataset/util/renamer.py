import os

root = "../patches/mask/trainB/"

files = os.listdir(root)
for i, file in enumerate(files):
    print(f'{i} out of {len(files)}')
    os.rename(root+file, (root+file).replace('.jpg', ''))