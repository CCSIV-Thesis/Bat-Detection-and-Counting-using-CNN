# Import libraries
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import imp
import scipy

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

#%%

# Define data path
PATH = os.getcwd()
data_path = PATH + '\\databats'
data_dir_list = os.listdir(data_path)

#Defining the rows,cols,channels,and epochs for Convolutional Layer inputs.
img_rows=64
img_cols=64
num_channel=3
num_epoch=20

#We initalized a list of image data and labels to be blank for the preprocessing part.
img_data_list=[]
labels_name = {'bat':0,'non-bat':1}
labels_list = []

#In this process, we use the Canny Edge Detection and the Watershed Algorithms to preprocess every image for both bats and non-bats.
#At the end of the process, every image should have markers
for dataset in data_dir_list:
	img_list=os.listdir(data_path+'\\'+ dataset)
	print ('Loading the images of dataset-'+'{}\n'.format(dataset))
	print ("Please wait...")
	label = labels_name[dataset]
	for img in img_list:
		input_img= cv2.imread(data_path + '\\'+ dataset + '\\'+ img)
		# Write image onto disk
		# Convert image from RGB to GRAY
		gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		# apply thresholding to convert the image to binary
		ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		# erode the image
		fg = cv2.erode(thresh, None, iterations=1)
		# Dilate the image
		bgt = cv2.dilate(thresh, None, iterations=1)
		# Apply thresholding
		ret, bg = cv2.threshold(bgt, 1, 128, 1)
		# Add foreground and background
		marker = cv2.add(fg, bg)
		# Apply canny edge detector
		canny = cv2.Canny(marker, 300, 350)
		# Finding the contors in the image using chain approximation
		new, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# converting the marker to int 32 bit
		marker32 = np.int32(marker)
		# Apply watershed algorithm
		cv2.watershed(input_img,marker32)
		m = cv2.convertScaleAbs(marker32)
		# Apply thresholding on the image to convert to binary image
		ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		# Invert the thresh
		thresh_inv = cv2.bitwise_not(thresh)
		# Bitwise and with the image mask thresh
		res = cv2.bitwise_and(input_img, input_img, mask=thresh)
		# Bitwise and the image with mask as threshold invert
		res3 = cv2.bitwise_and(input_img, input_img, mask=thresh_inv)
		# Take the weighted average
		res4 = cv2.addWeighted(res, 1, res3, 1, 0)
		# Draw the markers
		final = cv2.drawContours(res4, contours, -1, (0, 255, 0), 1)
		# Resize the preprocessed image into a 128x128 image
		input_img_resize=cv2.resize(final,(img_rows,img_cols))
		img_data_list.append(input_img_resize)
		labels_list.append(label)

cv2.imshow("Canny+Watershed", input_img_resize)
cv2.imshow("Grayscale", gray)
cv2.imshow("Foreground", fg)
cv2.imshow("Background", bg)
cv2.imshow("Marker", marker)
cv2.imshow("Thresh", res)
cv2.imshow("Thresh Inverse", res3)
cv2.imshow("Canny", canny)
cv2.resizeWindow("Canny+Watershed",280,280)
cv2.resizeWindow("Grayscale",280,280)
cv2.resizeWindow("Foreground",280,280)
cv2.resizeWindow("Background",280,280)
cv2.resizeWindow("Marker",280,280)
cv2.resizeWindow("Thresh",280,280)
cv2.resizeWindow("Thresh Inverse",280,280)
cv2.resizeWindow("Canny",280,280)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
#The reason that the values for each image vector should be normalized to 0-1 is that it would be easier for the model later on
#to process the values for classification.
img_data /= 255

num_classes = 2
labels = np.array(labels_list)
# print the count of number of samples for different classes
print(np.unique(labels,return_counts=True))
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)
#
#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=10)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)

#%%
# Defining the model
input_shape=img_data[0].shape
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
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=["accuracy"])
#
# # Viewing model_configuration
#
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable
#%%
# Training
hist = model.fit(X_train, y_train, batch_size=16, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))

# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])
plt.show()

# Evaluating the model

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

# Testing a new image
# Write image onto disk
test_image = cv2.imread('b.PNG')
# Convert image from RGB to GRAY
gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
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
cv2.watershed(test_image,marker32)
m = cv2.convertScaleAbs(marker32)
# Apply thresholding on the image to convert to binary image
ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Invert the thresh
thresh_inv = cv2.bitwise_not(thresh)
# Bitwise and with the image mask thresh
res = cv2.bitwise_and(test_image, test_image, mask=thresh)
# Bitwise and the image with mask as threshold invert
res3 = cv2.bitwise_and(test_image, test_image, mask=thresh_inv)
# Take the weighted average
res4 = cv2.addWeighted(res, 1, res3, 1, 0)
# Draw the markers
final = cv2.drawContours(res4, contours, -1, (0, 255, 0), 1)

test_image=cv2.resize(final,(img_rows,img_cols))
cv2.imshow("Final Test Image",test_image)
cv2.resizeWindow("Final Test Image",280,280)
# cv2.waitKey(0)
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
# print (test_image.shape)
test_image= np.expand_dims(test_image, axis=0)
print (test_image.shape)

# Predicting the test image
print((model.predict(test_image)))
print(model.predict_classes(test_image))

# Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
target_names = ['class 0(Bats)', 'class 1(Non-Bats)']

print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


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

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
plt.show()

#%%
# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

model.save('model.model')
loaded_model=load_model('model.model')
