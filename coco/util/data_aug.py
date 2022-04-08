import albumentations as A
import cv2 as cv
import os
import random

movie_root = '../../references/movie_references/'
movie_transform = A.Compose([
    A.GaussNoise(always_apply=True),
    A.GaussianBlur(always_apply=True),
    A.HistogramMatching(
        random.sample([movie_root+x for x in os.listdir(movie_root)], k=500),
        always_apply=True
    ),
])

game_root = '../../references/game_references/'
game_transform = A.Compose([
    A.CLAHE(always_apply=True),
    A.FDA(
        random.sample([game_root+x for x in os.listdir(game_root)], k=500), 
        beta_limit=0.01, 
        always_apply=True
    )
])

transform = A.Compose([
    A.OneOf([
        movie_transform,
        game_transform,
    ], p=0.5)
])


root = '../filtered_images/val/'
random_image = random.choice(os.listdir(root))
img = cv.imread(root+random_image)

transformed_img = movie_transform(image=img)['image']
cv.imwrite('transformed.jpg', transformed_img)


