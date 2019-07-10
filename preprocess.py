from skimage.io import imread, imsave
from skimage.transform import resize
from scipy.io import loadmat
from tqdm import tqdm

data = loadmat('cars_annos.mat')
img_info = data['annotations'][0]

img_files = []
img_classes = []
cc, image_idx = 0, 1
for info in img_info:
    if info[-2][0][0] == image_idx:
        img_files.append(info[0][0])
        img_classes.append(str(info[-2][0][0]))
        cc += 1
    else:
        continue
    if cc == 6:
        cc = 0
        image_idx += 1

img_final = [img.replace('ims', 'imgs') for img in img_files]

for source, dest in tqdm(zip(img_files, img_final)):
    img = imread(source)
    img2 = resize(img, (300, 400), anti_aliasing=True)
    imsave(dest, img2)

with open('img_cars.csv', 'w') as writer:
    writer.write('Image, Car\n')
    writer.write('\n'.join([img + ',' + idx for img, idx in zip(img_final, img_classes)]))
