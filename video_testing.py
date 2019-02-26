import cv2
import numpy as np
from keras.models import load_model

#Initializing the bat counter, the output video, and the model
batCounter = 0
finalCount = 0
z = 0
vid = cv2.VideoCapture('batsflyout7.mp4')
model = load_model("newBat.model")

def preprocessing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    cv2.watershed(frame,marker32)
    # Apply watershed algorithm
    m = cv2.convertScaleAbs(marker32)
    ret, thresh = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Apply thresholding on the image to convert to binary image
    thresh_inv = cv2.bitwise_not(thresh)
    # Invert the thresh
    res = cv2.bitwise_and(frame, frame, mask=thresh)
    # Bitwise and with the image mask thresh
    res3 = cv2.bitwise_and(frame, frame, mask=thresh_inv)
    # Bitwise and the image with mask as threshold invert
    res4 = cv2.addWeighted(res, 1, res3, 1, 0)
    # Take the weighted average

    final = cv2.drawContours(res4, contours, -1, (0, 255, 0), 1)
    # print("Preprocessed image shape: ",final.shape)
    # cv2.imwrite("Final img.png",final)
    return final

def predictions(frame,batCounter,direction):
    IMG_SIZE = 64
    num_channel = 3
    if(direction == "1"): #Right
        x = frame.shape[1] - 150
        y = frame.shape[0]
    elif(direction == "2"): #Top
        x = frame.shape[1]
        y = frame.shape[0] - 276 #570
    elif(direction == "3"): #Left
        x = frame.shape[1] - 490    #1130
        y = frame.shape[0]
    elif(direction == "4"): #Bottom
        x = frame.shape[1]
        y = frame.shape[0] - 150
    point = 0
    if(direction == "1" or direction == "3"):
        # print("wait")
        # print("X: ",x)
        while(point < y):
            if(direction == "1"):
                crop_img = frame[point:point+IMG_SIZE,x:x+IMG_SIZE]
            else:
                crop_img = frame[point:point+IMG_SIZE,x:x-IMG_SIZE]
            # print("Cropped image shape: ",crop_img.shape)
            # print("Current point: ",point)
            # print("MAX Y: ",y)
            cv2.imshow("frame",frame)
            cv2.imshow("cropped",crop_img)
            cv2.waitKey(100)
            # cv2.destroyAllWindows()
            # cv2.waitKey(1)
            if(crop_img.shape[0] != IMG_SIZE):
                zeros = np.zeros((IMG_SIZE - crop_img.shape[0])*IMG_SIZE*num_channel,dtype="uint8").reshape(((IMG_SIZE - crop_img.shape[0]),IMG_SIZE,num_channel))
                crop_img =  np.concatenate((crop_img,zeros))
            crop_img = crop_img.astype('float32')
            crop_img /= 255
            shape_predict = crop_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            prediction = model.predict([shape_predict])
            # pred_batch = model.predict_on_batch([shape_predict])
            # print(pred_batch)
            print(prediction)
            if(prediction[0][0] > prediction[0][1]):
                print("Bat")
                batCounter = batCounter + 1
            else:
                print("Non-bat")
            point = point + 64
    elif(direction == "2" or direction == "4"):
        while(point < x):
            if(direction == "2"):
                crop_img = frame[y:y+IMG_SIZE,point:point-IMG_SIZE]
            else:
                crop_img = frame[y:y+IMG_SIZE,point:point+IMG_SIZE]
            # print("Cropped image shape: ",crop_img.shape)
            # print("Current point: ",point)
            # print("MAX Y: ",y)
            cv2.imshow("cropped",crop_img)
            cv2.waitKey(1000)
            if(crop_img.shape[0] != IMG_SIZE):
                zeros = np.zeros((IMG_SIZE - crop_img.shape[0])*IMG_SIZE*num_channel,dtype="uint8").reshape(((IMG_SIZE - crop_img.shape[0]),IMG_SIZE,num_channel))
                crop_img =  np.concatenate((crop_img,zeros))
            crop_img = crop_img.astype('float32')
            crop_img /= 255
            shape_predict = crop_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            prediction = model.predict([shape_predict])
            # print(shape_predict)
            print(prediction)
            if(prediction[0][0] > prediction[0][1]):
                print("Bat")
                batCounter = batCounter + 1
            else:
                print("Non-bat")
            point = point + 64
    # print("Bat Count for this Frame: ", batCounter)
    return batCounter
print("Which direction does the bats go?\n1 for Right \n2 for Top \n3 for Left \n4 for Down\n")
direction = input()
print("Processing....")
while (vid.isOpened()):
    if(z < vid.get(cv2.CAP_PROP_FRAME_COUNT)):
        ret, frame = vid.read()
        preprocessedFrame = preprocessing(frame)
        frameCount = predictions(preprocessedFrame,batCounter,direction)
        finalCount = finalCount + frameCount
        z = z + 1
    else:
        break

print("Final Bat Count for the Entire Video: ", finalCount)















# print("Y:",y)
# print("X:",x)
# li = cv2.line(img=frame, pt1=(x, 0), pt2=(x, y), color=(255, 255, 255), thickness=1, lineType=8, shift=0)
# cv2.imshow("Preprocessed with line",li)

# print("Y:",y)
# print("X:",x)
# li = cv2.line(img=frame, pt1=(0, y), pt2=(x, y), color=(255, 255, 255), thickness=1, lineType=8, shift=0)
# cv2.imshow("Preprocessed with l ine",li)
# cv2.waitKey(0)
