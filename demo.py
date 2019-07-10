from skimage.io import imread_collection
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

df = pd.read_csv('img_cars.csv')
y = df.iloc[:,1].values

imgs = imread_collection('car_imgs/*')
p_x, p_y = imgs[0].shape[:2]

X = []

for img in tqdm(imgs):
    pixels = []
    for i in range(p_x):
        for j in range(p_y):
            if len(img.shape) == 3:
                pixels.extend(img[0,0,:])
            else:
                pixels.extend([img[0,0]]*3)
    X.append(pixels)

X = np.array(X)

np.savez_compressed('cars_scaled.npz', X=X, y=y)
