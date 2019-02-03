import cv2
import numpy as np
from keras.models import load_model

test_image = cv2.imread("images.jpg")
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
canny = cv2.Canny(marker, 10, 15)
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
final_img = cv2.resize(final,(128,128))
print(final_img.shape)
cv2.imshow("Final Image",final_img)

test_img = cv2.imread("images.jpg")
crop_img = test_img[0:128, 0:128]
cv2.imshow("original",test_img)
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

model = load_model("model.model")
out = cv2.VideoCapture('output.mp4')
batCounter = 0
frameCount = out.get(cv2.CAP_PROP_FRAME_COUNT)
print("Please Wait.....")
print("Frame count: ", out.get(cv2.CAP_PROP_FRAME_COUNT))

def prepare(filepath):
    IMG_SIZE = 64  # 50 in txt-based
    # img_array = cv2.imread(filepath)
    # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    # cv2.imshow("Test Image", new_array)
    # cv2.waitKey(0)
    # print(new_array.shape)
    # Testing a new image
    test_image = cv2.imread(filepath)
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
    test_image=cv2.resize(final,(IMG_SIZE,IMG_SIZE))
    cv2.imshow("Final Test Image",test_image)
    cv2.waitKey(0)
    return test_image.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

prediction = model.predict([prepare("sunset.jpg")])
print(prediction[0])
if(prediction[0][0] > prediction[0][1]):
    batCounter = batCounter + 1


# while (out.isOpened()):
#     # Capture frame-by-frame
#     ret, frame = out.read()
#     # cv2.imshow("Frame",frame)
#     # cv2.waitKey(0)
#     # predict = prepare(frame)


print("Bat Count: ", batCounter)
