import pandas
import numpy as np
import PIL.Image as Image
import pickle
from keras.models import load_model
from keras.models import Sequential

metadata_file = './covid-chestxray-dataset-master/metadata.csv'
images_path = './covid-chestxray-dataset-master/images/'

def store_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=4)

def get_data(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

def get_filenames():
    df = pandas.read_csv(metadata_file)
    covidLines = np.array(df["finding"]) == 'COVID-19'
    covidImageFiles = df["filename"][covidLines]

    return covidImageFiles

def load_images(files, transformation, size, rotation):
    list_images = []
    not_found = 0

    for f in files:
        try:
            with Image.open(images_path + f, 'r') as image:
                imageNormal = transformation(image, size, 0)
                list_images.append(np.array(imageNormal))
                # imagePos = transformation(image, size, rotation)
                # list_images.append(np.array(imagePos))
                # imageNeg = transformation(image, size, -rotation)
                # list_images.append(np.array(imageNeg))
        except:
            not_found += 1

    print("There was", not_found, "images not found")

    X = np.array(list_images)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))

    y = np.ones(X.shape[0], dtype=int)

    # One hot vectors
    Y = np.zeros((2, y.size))
    Y[y, np.arange(y.size)] = 1

    return X, y, Y

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

def App1(image, size, rotation):
    image = crop_excess(image, size)
    image = image.resize(size, Image.BILINEAR)
    image = image.rotate(rotation)
    image = image.convert(mode='L')
    return image

if __name__ == "__main__":
    print("Loading dataset...")
    if True:
        files = get_filenames()
        X, y, Y = load_images(files, App1, (512, 512), 15)

        test = get_data("test_app1.p")

        testX = test["X"]
        testY = test["Y"]
        test_y = test["y"]

        normal_samples = (test_y == 0)
        testX = testX[normal_samples]
        testY = testY[:, normal_samples]
        test_y = test_y[normal_samples]

        # print(Y.shape)
        # print(testY.shape)
        X = np.concatenate((X, testX), axis=0)
        Y = np.concatenate((Y, testY), axis=1)
        y = np.concatenate((y, test_y))

        # X = testX
        # Y = testY
        # y = test_y

        covid = {'X': X, 'y': y, 'Y': Y}
        store_object(covid, "covid_app1.p")
    else:
        covid = get_data("test_app1.p")

    print("Loaded")
    X = covid["X"]
    Y = covid["Y"].T
    y = covid["y"]

    model = load_model("model_conv.hdf5")
    acc = model.evaluate(X, Y)[1]

    model = load_model("model_conv_retrained.hdf5")
    acc1 = model.evaluate(X, Y)[1]
    
    print("Model without weigths:", acc, "\nModel with weights:", acc1)
