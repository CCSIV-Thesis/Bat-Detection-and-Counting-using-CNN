# USAGE
# python image_training.py --dataset datasetname --model modelname.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
from sklearn.metrics import classification_report,confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
img_rows=64
img_cols=64
# epochs = 150
initial_learning = 1e-3
batch_size = 1

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	# Convert image from RGB to GRAY
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# apply thresholding to convert the image to binary
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# erode the image
	fg = cv2.erode(thresh, None, iterations=1)
	# Dilate the image
	bgt = cv2.dilate(thresh, None, iterations=1)
	# Apply thresholding
	ret, bg = cv2.threshold(bgt, 1, 64, 1)
	# Add foreground and background
	marker = cv2.add(fg, bg)
	# Apply canny edge detector
	canny = cv2.Canny(marker, 300, 350)
	# Finding the contors in the image using chain approximation
	new, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# converting the marker to float 32 bit
	marker32 = np.int32(marker)
	# Apply watershed algorithm
	cv2.watershed(image,marker32)
	m = cv2.convertScaleAbs(marker32)
	# Apply thresholding on the image to convert to binary image
	ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# Invert the thresh
	thresh_inv = cv2.bitwise_not(thresh)
	# Bitwise and with the image mask thresh
	res = cv2.bitwise_and(image, image, mask=thresh)
	# Bitwise and the image with mask as threshold invert
	res3 = cv2.bitwise_and(image, image, mask=thresh_inv)
	# Take the weighted average
	res4 = cv2.addWeighted(res, 1, res3, 1, 0)
	# Draw the markers
	final = cv2.drawContours(res4, contours, -1, (0, 255, 0), 1)
	image = cv2.resize(final,(img_rows,img_cols))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "bat" else 0
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
# Defining the model
num_classes = 2
input_shape=data[0].shape
print(input_shape)
model = Sequential()
model.add(Conv2D(16,kernel_size=(3,3),padding='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(16,kernel_size=(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(32,kernel_size=(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32,kernel_size=(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
opt = Adam(lr=initial_learning, decay=initial_learning)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# # Viewing model_configuration
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# train the network
print("[INFO] training network...")
fit = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch_size,
	epochs=1000, verbose=1)

es_epochs = len(history.history['loss'])

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("classic")
plt.figure()
N = es_epochs
plt.plot(np.arange(0, N), fit.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), fit.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), fit.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), fit.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Bat/Non Bat")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
plt.show()

# Printing the confusion matrix

Y_pred = model.predict(testX)
y_pred = np.argmax(Y_pred, axis=1)
target_names = ['class 0(Bats)', 'class 1(Non-Bat)']

print(classification_report(np.argmax(testY,axis=1), y_pred,target_names=target_names))

# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(testY,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
plt.savefig("conf_matrix")
plt.show()
