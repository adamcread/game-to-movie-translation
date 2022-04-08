from PIL import Image
import os

root = '../filtered_images/val/'
dest = '../converted_filtered/val/'
images = os.listdir(root)

for i, image in enumerate(images):
    print(f'{i} out of {len(images)}')
    file_name = image.split('.')[0]
    im = Image.open(root+image)

    im.save(dest+file_name+'.png')