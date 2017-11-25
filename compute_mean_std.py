import mxnet as mx
from mxnet import nd
import numpy as np
import os

path = './data/ai_challenger_scene_train_20170904/scene_train_images_20170904/'

img_list = os.listdir(path)

r = 0 # r mean
g = 0 # g mean
b = 0 # b mean

r_2 = 0 # r^2 
g_2 = 0 # g^2
b_2 = 0 # b^2

total = 0
for img_name in img_list:
    img = mx.image.imread(path + img_name) # ndarray, width x height x 3
    img = img.astype('float32') / 255.
    total += img.shape[0] * img.shape[1]
    
    r += img[:, :, 0].sum().asscalar()
    g += img[:, :, 1].sum().asscalar()
    b += img[:, :, 2].sum().asscalar()
    
    r_2 += (img[:, :, 0]**2).sum().asscalar()
    g_2 += (img[:, :, 1]**2).sum().asscalar()
    b_2 += (img[:, :, 2]**2).sum().asscalar()

r_mean = r / total
g_mean = g / total
b_mean = b / total

r_var = r_2 / total - r_mean ** 2
g_var = g_2 / total - g_mean ** 2
b_var = b_2 / total - b_mean ** 2

print('mean r: {} g: {}, b: {}'.format(r_mean, g_mean, b_mean))
print('var r: {}, g: {}, b: {}'.format(np.sqrt(r_var), np.sqrt(g_var), np.sqrt(b_var)))

# mean r: 0.49603017947930517 g: 0.47806492768627135, b: 0.4476716685422784
# var r: 0.29148932673400635, g: 0.2863660174073978, b: 0.29812326471339234


