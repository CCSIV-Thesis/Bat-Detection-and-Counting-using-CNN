# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

img_rows=64
img_cols=64

# load the image
image = cv2.imread(args["image"])
if image is None:
	print("No image passed")
else:
	orig = image.copy()

# pre-process the image for classification
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
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# classify the input image
(non_bat, bat) = model.predict(image)[0]

# build the label
label = "Bat" if bat > non_bat else "Non Bat"
proba = bat if bat > non_bat else non_bat
label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
# layer_outputs = [layer.output for layer in model.layers[:10]] # Extracts the outputs of the top 10 layers
# activation_model = model.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
# activations = activation_model.predict(test_image) # Returns a list of five Numpy arrays: one array per layer activation
#
# # first_layer_activation = activations[0]
# # print(layer_outputs)
# # print(activations.shape)
# # print(first_layer_activation.shape)
# # plt.matshow(first_layer_activation[0, :, :, 0], cmap='gray')
# # plt.title(first_layer_activation)
# # plt.show()
#
# layer_name = []
# num = 0
# ch = 0
# for layer in model.layers:
#     layer_name.append(layer.name) # Names of the layers, so you can have them as part of your plot
# for name in layer_name: # Displays the feature maps
# 	layer_outputs = name #get layer name
# 	activate = activations[num] #get activation layer
# 	num = num + 1
# 	print(name) #prints the name of layer
# 	print(activate.shape) #prints the shape of the activations
# 	while ch < 3: #controls the different channels
# 		plt.matshow(activate[0, :, :, ch])
# 		title = name + " channel " + str(ch)
# 		plt.title(title)
# 		print("channel ", ch)
# 		plt.show()
# 		ch = ch + 1
# 	ch = 0
# 	if num == 10:
# 		break
#
