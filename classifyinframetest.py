import cv2
import numpy as np

def prepare(filepath,frameCount):
    if i < frameCount:
        # test_image = cv2.imread(filepath)
        # Write image onto disk
        # cv2.line(img=filepath, pt1=(100, 0), pt2=(100, 1000000), color=(255, 255, 255), thickness=5, lineType=8, shift=0)
        gray = cv2.cvtColor(filepath, cv2.COLOR_BGR2GRAY)
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
        cv2.watershed(filepath,marker32)
        # Apply watershed algorithm
        m = cv2.convertScaleAbs(marker32)
        ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Apply thresholding on the image to convert to binary image
        thresh_inv = cv2.bitwise_not(thresh)
        # Invert the thresh
        res = cv2.bitwise_and(filepath, filepath, mask=thresh)
        # Bitwise and with the image mask thresh
        res3 = cv2.bitwise_and(filepath, filepath, mask=thresh_inv)
        # Bitwise and the image with mask as threshold invert
        res4 = cv2.addWeighted(res, 1, res3, 1, 0)
        # Take the weighted average
        final = cv2.drawContours(res4, contours, -1, (0, 255, 0), 1)
        test_image=cv2.resize(final,(IMG_SIZE,IMG_SIZE))
        # cv2.imshow("Final Test Image",final)
        # cv2.waitKey(0)
        # print(final.shape)
        return test_image.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    else:
        print(filepath)

model = load_model("model.model")
out = cv2.VideoCapture('output.mp4')
batCounter = 0
frameCount = out.get(cv2.CAP_PROP_FRAME_COUNT)
print("Please Wait.....")
print("Frame count: ", out.get(cv2.CAP_PROP_FRAME_COUNT))
while (out.isOpened()):
    # Capture frame-by-frame
    ret, frame = out.read()
    # cv2.imshow("Frame",frame)
    # cv2.waitKey(0)
    # predict = prepare(frame)
    prediction = model.predict([prepare(frame,frameCount)])
    # print(prediction[0][0])
    if(prediction[0][0] > prediction[0][1]):
        batCounter = batCounter + 1

print("Bat Count: ", batCounter)
