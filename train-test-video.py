import cv2
import tensorflow as tf

CATEGORIES = ["Bat", "Non-Bat"]


def prepare(filepath):
    IMG_SIZE = 128  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


model = tf.keras.models.load_model("model2.model")

prediction = model.predict([prepare('images.jpg')])
# prediction = model.predict('images.jpg')
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])
