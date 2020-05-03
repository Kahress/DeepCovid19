import sys
import os
from PIL import Image
from PIL import ImageStat
import numpy as np
import pickle

dataset_path = '.\\chest_xray\\'

'''
with Image.open('.\\chest_xray\\train\\NORMAL\\IM-0115-0001.jpeg', 'r') as image:
    new_image = image.resize(min_gen)

image.show()
new_image.show()
'''

# np.array(im)

# 1. To smallest size
# 2. To largest size
# 3. To average size
# 4. To constant values (w, h)

def compute_sizes():
    num_img = 0
    min_size = (1000000,1000000)
    max_size = (0,0)
    avg_w = 0
    avg_h = 0
    avg_size = (0,0)
    
    for dirpath, dirnames, files in os.walk(dataset_path + 'train'):
        for f in files:
            with Image.open(dirpath + "\\" + f, 'r') as image:
                num_img += 1
                w, h = image.size

                # Min size calculation

                if w*h < min_size[0]*min_size[1]:
                    min_size = (w,h)

                # Max size calculation

                if w*h > max_size[0]*max_size[1]:
                    max_size = (w,h)

                # Average size calculation

                avg_w += w
                avg_h += h
                
    avg_size = (int(avg_w/num_img), int(avg_h/num_img))

    return min_size, max_size, avg_size

def load_images(path, transformation, size):
    list_normal = []
    list_pneumonia = []

    for dirpath, _, files in os.walk(path + '\\NORMAL'):
        for f in files:
            with Image.open(dirpath + "\\" + f, 'r') as image:
                image = transformation(image, size)
                list_normal.append(np.array(image))
         
    for dirpath, _, files in os.walk(path + '\\PNEUMONIA'):
        for f in files:
            with Image.open(dirpath + "\\" + f, 'r') as image:
                image = transformation(image, size)
                list_pneumonia.append(np.array(image))

    X = np.concatenate((np.array(list_normal), np.array(list_pneumonia)))

    y = np.concatenate((np.zeros(len(list_normal), dtype=int), np.ones(len(list_pneumonia), dtype=int)))

    Y = np.zeros((2, y.size))
    Y[y, np.arange(y.size)] = 1

    return X, y, Y

def create_dataset(transformation = (lambda x, y: x), size = (0,0)) :

    train_path = dataset_path + 'train'
    val_path = dataset_path + 'val'
    test_path = dataset_path + 'test'

    print("Loading train...")
    X, y, Y = load_images(train_path, transformation, size)
    train = {'X': X, 'y': y, 'Y': Y}

    print("Loading validation...")
    X, y, Y = load_images(val_path, transformation, size)
    val = {'X': X, 'y': y, 'Y': Y}

    print("Loading test...")
    X, y, Y = load_images(test_path, transformation, size)
    test = {'X': X, 'y': y, 'Y': Y}
    
    dataset = {'train': train, 'val': val, 'test': test}

    return dataset

def store_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

############################
######### Resizing #########
############################

def resize_to_size(image, size):
    return image.resize(size)

############################
######### Cropping #########
############################

def crop_to_size(image, size):
    im_w, im_h = image.size
    crop_w, crop_h = size
    # Middle area
    left_side = (im_w - crop_w)/2
    right_side = left_side + crop_w
    top_side = (im_h - crop_h)/2
    bottom_side = top_side + crop_h
    
    return image.crop((left_side, top_side, right_side, bottom_side))

############################
######### Excesses #########
############################

def crop_excess(image, size):
    im_w, im_h = image.size
    w, h = size
    wth_ratio = w / h
    image_ratio = im_w / im_h

    if image_ratio > wth_ratio:
        # Width cropping
        crop = (im_w - im_h*wth_ratio) / 2
        left_side = crop
        right_side = im_w - crop
        top_side = 0
        bottom_side = im_h
    else:
        # Heigth cropping or none
        crop = (im_h - (im_w / wth_ratio)) / 2
        left_side = 0
        right_side = im_w
        top_side = crop
        bottom_side = im_h - crop

    return image.crop((left_side, top_side, right_side, bottom_side))

def pad_excess(image, size):
    im_w, im_h = image.size
    w, h = size
    wth_ratio = w / h
    image_ratio = im_w / im_h

    if image_ratio > wth_ratio:
        # Width cropping
        crop = ((im_w / wth_ratio) - im_h) / 2
        left_side = 0
        right_side = im_w
        top_side = - crop
        bottom_side = im_h + crop
    else:
        # Heigth cropping or none
        crop = (im_h*wth_ratio - im_w) / 2
        left_side = - crop
        right_side = im_w + crop
        top_side = 0
        bottom_side = im_h

    return image.crop((left_side, top_side, right_side, bottom_side))


############################
##### Paper approaches #####
############################

############################
######## Approach 1 ########
############################

def App1(image):
    size = (1920, 1080)
    image.show()

    print('Cropping excess')
    image = crop_excess(image, size)
    image.show()

    print('Resizing')
    image = image.resize(size, Image.BILINEAR)
    image.show()

    return image

############################
######## Approach 2 ########
############################

def App2(image):
    print('Computing size')
    min_gen, max_gen, avg_gen = compute_sizes()
    size = max_gen

    print('Cropping to largest')
    image = crop_to_size(image, size)
    image.show()

    return image

############################
##### Common Approach ######
############################

def OTSU(image, threshold):
    return image.point(lambda p: p > threshold and p)

def Normalization(image):
    stats = ImageStat.Stat(image)
    std = stats.stddev[0]

    return image.point(lambda p: p / std)

def CommonApp(image):
    image = OTSU(image, 100)
    image.show()
    image = Normalization(image)

    return image


if __name__ == "__main__":

    debug = True

    if debug:

        for dirpath, dirnames, files in os.walk(dataset_path + 'train'):
            for f in files:
                with Image.open(dirpath + "\\" + f, 'r') as image:
                    image = App1(image)
                break
        exit()

    print("Computing sizes...")
    min_gen, max_gen, avg_gen = compute_sizes()

    print("Creating dataset...")
    dataset = create_dataset()

    store_object(dataset, "dataset.p")