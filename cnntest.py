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

PATH = os.getcwd()
# Define data path
data_path = PATH + '\\databats'
data_dir_list = os.listdir(data_path)

img_rows=128
img_cols=128
num_channel=3
num_epoch=20

img_data_list=[]
labels_name = {'bat':0,'non-bat':1}
labels_list = []

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'\\'+ dataset)
	print ('Loading the images of dataset-'+'{}\n'.format(dataset))
	print ("Please wait...")
	label = labels_name[dataset]
	for img in img_list:
		input_img= cv2.imread(data_path + '\\'+ dataset + '\\'+ img)
		# Write image onto disk
		gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		# Convert image from RGB to GRAY
		ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		# apply thresholding to convert the image to binary
		fg = cv2.erode(thresh, None, iterations=1)
		# erode the image
		bgt = cv2.dilate(thresh, None, iterations=1)
		# Dilate the image
		ret, bg = cv2.threshold(bgt, 1, 128, 1)
		# Apply thresholding
		marker = cv2.add(fg, bg)
		# Add foreground and background
		canny = cv2.Canny(marker, 300, 350)
		# Apply canny edge detector
		new, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# Finding the contors in the image using chain approximation
		marker32 = np.int32(marker)
		# converting the marker to float 32 bit
		cv2.watershed(input_img,marker32)
		# Apply watershed algorithm
		m = cv2.convertScaleAbs(marker32)
		ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		# Apply thresholding on the image to convert to binary image
		thresh_inv = cv2.bitwise_not(thresh)
		# Invert the thresh
		res = cv2.bitwise_and(input_img, input_img, mask=thresh)
		# Bitwise and with the image mask thresh
		res3 = cv2.bitwise_and(input_img, input_img, mask=thresh_inv)
		# Bitwise and the image with mask as threshold invert
		res4 = cv2.addWeighted(res, 1, res3, 1, 0)
		# Take the weighted average
		final = cv2.drawContours(res4, contours, -1, (0, 255, 0), 1)
		input_img_resize=cv2.resize(final,(128,128))
		img_data_list.append(input_img_resize)
		labels_list.append(label)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
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
model.add(Conv2D(32,kernel_size=(3,3),padding='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32,kernel_size=(3,3),padding='same'))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="tf"))
model.add(Dropout(0.5))

model.add(Conv2D(64,kernel_size=(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,kernel_size=(3,3),padding='same'))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="tf"))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
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

print(Y)
#%%
# Training
hist = model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=1, validation_data=(X_test, y_test))

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
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()
# #%%
#
# # Evaluating the model
#
# score = model.evaluate(X_test, y_test, verbose=0)
# print('Test Loss:', score[0])
# print('Test accuracy:', score[1])
#
# test_image = X_test[0:1]
# # X_test = X_test * 255
# # X_test = X_test.astype('int64')
# # print(X_test[0:1])
# # print(X_test[0:1].shape)
# # cv2.imshow("Test Image", X_test[0:1])
# # cv2.waitKey(0)
# print(model.predict(test_image))
# print(model.predict_classes(test_image))
# print(y_test[0:1])
#
# Testing a new image
test_image = cv2.imread('scolopax_rochussenii_m_1.jpg')
# Write image onto disk
gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
# Convert image from RGB to GRAY
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# apply thresholding to convert the image to binary
fg = cv2.erode(thresh, None, iterations=1)
# erode the image
bgt = cv2.dilate(thresh, None, iterations=1)
# Dilate the image
ret, bg = cv2.threshold(bgt, 1, 128, 1)
# Apply thresholding
marker = cv2.add(fg, bg)
# Add foreground and background
canny = cv2.Canny(marker, 300, 350)
# Apply canny edge detector
new, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Finding the contors in the image using chain approximation
marker32 = np.int32(marker)
# converting the marker to float 32 bit
cv2.watershed(test_image,marker32)
# Apply watershed algorithm
m = cv2.convertScaleAbs(marker32)
ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Apply thresholding on the image to convert to binary image
thresh_inv = cv2.bitwise_not(thresh)
# Invert the thresh
res = cv2.bitwise_and(test_image, test_image, mask=thresh)
# Bitwise and with the image mask thresh
res3 = cv2.bitwise_and(test_image, test_image, mask=thresh_inv)
# Bitwise and the image with mask as threshold invert
res4 = cv2.addWeighted(res, 1, res3, 1, 0)
# Take the weighted average
final = cv2.drawContours(res4, contours, -1, (0, 255, 0), 1)
test_image=cv2.resize(final,(128,128))
cv2.imshow("Final Test Image",test_image)
cv2.waitKey(0)
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
# print (test_image.shape)
test_image= np.expand_dims(test_image, axis=0)
print (test_image.shape)

# Predicting the test image
print((model.predict(test_image)))
print(model.predict_classes(test_image))
#
# #%%
#
# Visualizing the intermediate layer

#
def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations

layer_num=3
filter_num=0

activations = get_featuremaps(model, int(layer_num),test_image)

print (np.shape(activations))
feature_maps = activations[0][0]
print (np.shape(feature_maps))

if K.image_dim_ordering()=='th':
	feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
print (feature_maps.shape)

fig=plt.figure(figsize=(16,16))
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.jpg')

num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(16,16))
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
	ax = fig.add_subplot(subplot_num, subplot_num, i+1)
	#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
	ax.imshow(feature_maps[:,:,i],cmap='gray')
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
plt.show()
fig.savefig("featuremaps-layer-{}".format(layer_num) + '.jpg')

#%%
# Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
#y_pred = model.predict_classes(X_test)
#print(y_pred)
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
#plt.figure()
# Plot normalized confusion matrix
#plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                      title='Normalized confusion matrix')
#plt.figure()
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

model.save('model.hdf5')
loaded_model=load_model('model.hdf5')
