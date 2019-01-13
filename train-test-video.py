import cv2
import tensorflow as tf
import numpy as np

from keras.models import load_model

CATEGORIES = ["Bat", "Not-Bat"]


def prepare(filepath):
    IMG_SIZE = 128  # 50 in txt-based
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    print(new_array.shape)
    # new_array = np.expand_dims(new_array, axis=)
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


model = load_model("model2.model")

prediction = model.predict([prepare('batsflyout7.mp4')])
# prepare('images2.jpg')
print(prediction)  # will be a list in a list.
# print(CATEGORIES[int(prediction[0][1])])
