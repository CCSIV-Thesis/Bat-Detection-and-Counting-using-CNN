import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix
from keras.utils import to_categorical

#Initializing the bat counter, the output video, and the model
batCounter = 0
finalCount = 0
total_estimated = 0
z = 0
threshold = .8
vid = cv2.VideoCapture('output3.mp4')
model = load_model("anAwesome.model")

def predictions(frame,batCounter,direction):
    IMG_SIZE = 64
    num_channel = 3
    if(direction == "1"): #Right
        x = frame.shape[1] - 150
        y = frame.shape[0]
        li = cv2.line(img=frame, pt1=(x, 0), pt2=(x, y), color=(255, 255, 255), thickness=1, lineType=8, shift=0)
        cv2.imshow("Preprocessed with line",li)
    elif(direction == "2"): #Top
        x = frame.shape[1]
        y = frame.shape[0] - 570 #276
        li = cv2.line(img=frame, pt1=(0, y), pt2=(x, y), color=(255, 255, 255), thickness=1, lineType=8, shift=0)
        cv2.imshow("Preprocessed with line",li)
        # cv2.waitKey(0)
    elif(direction == "3"): #Left
        x = frame.shape[1] - 1130 #490
        y = frame.shape[0]
        li = cv2.line(img=frame, pt1=(x, 0), pt2=(x, y), color=(255, 255, 255), thickness=1, lineType=8, shift=0)
        cv2.imshow("Preprocessed with line",li)
    elif(direction == "4"): #Bottom
        x = frame.shape[1]
        y = frame.shape[0] - 150
        li = cv2.line(img=frame, pt1=(0, y), pt2=(x, y), color=(255, 255, 255), thickness=1, lineType=8, shift=0)
        cv2.imshow("Preprocessed with line",li)
        # cv2.waitKey(0)
    point = 0
    if(direction == "1" or direction == "3"):
        # print("X: ",x)
        while(point < y):
            if(direction == "1"):
                crop_img = frame[point:point+IMG_SIZE,x:x+IMG_SIZE]
            else:
                crop_img = frame[point:point+IMG_SIZE,x-IMG_SIZE:x]
            print(crop_img.shape)
            y_point = 0
            y64 = crop_img.shape[0]
            cv2.imshow("cropped",crop_img)
            cv2.waitKey(100)
            if(y64 == IMG_SIZE):
                while(y_point < y64):
                    x_point = 0
                    x64 = crop_img.shape[1]
                    while(x_point < x64):
                        pred_img = crop_img[y_point:y_point+16,x_point:x_point+16]
                        pred_img = pred_img.astype('float32')
                        pred_img /= 255.0
                        shape_predict = crop_img.reshape(-1, 16, 16, 3)
                        prediction = model.predict([shape_predict])
                        if(prediction[0][0] > prediction[0][1]):
                            print("Bat")
                            batCounter = batCounter + 1
                        else:
                            print("Non-Bat")
                        #50% sliding window
                        x_point = x_point + 8
                    y_point = y_point + 8
            # if(crop_img.shape[0] != IMG_SIZE):
            #     zeros = np.zeros((IMG_SIZE - crop_img.shape[0])*IMG_SIZE*num_channel,dtype="uint8").reshape(((IMG_SIZE - crop_img.shape[0]),IMG_SIZE,num_channel))
            #     crop_img =  np.concatenate((crop_img,zeros))
            # crop_img = crop_img.astype('float32')
            # crop_img /= 255.0
            # shape_predict = crop_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            # prediction = model.predict([shape_predict])
            # print(prediction)
            # if(prediction[0][0] > prediction[0][1]):
            #     print("Bat")
            #     batCounter = batCounter + 1
            # else:
            #     print("Non-Bat")
            point = point + 32
    elif(direction == "2" or direction == "4"):
        while(point < x):
            if(direction == "2"):
                crop_img = frame[y-IMG_SIZE:y,point:point+IMG_SIZE]
            else:
                crop_img = frame[y:y+IMG_SIZE,point:point+IMG_SIZE]
            cv2.imshow("cropped",crop_img)
            cv2.waitKey(100)
            if(crop_img.shape[0] != IMG_SIZE):
                zeros = np.zeros((IMG_SIZE - crop_img.shape[0])*IMG_SIZE*num_channel,dtype="uint8").reshape(((IMG_SIZE - crop_img.shape[0]),IMG_SIZE,num_channel))
                crop_img =  np.concatenate((crop_img,zeros))
            crop_img = crop_img.astype('float32')
            crop_img /= 255
            shape_predict = crop_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            prediction = model.predict([shape_predict])
            print(prediction)
            if(prediction[0][0] > prediction[0][1]):
                print("Bat")
                batCounter = batCounter + 1
            else:
                print("Non-Bat")
            #50% sliding window
            point = point + 32
    return batCounter

print("Total Estimated Count of Bats: ")
total_estimated = int(input())
print("Which direction does the bats go?\n1 for Right \n2 for Top \n3 for Left \n4 for Down\n")
direction = input()
print("Processing....")
while (vid.isOpened()):
    if(z < vid.get(cv2.CAP_PROP_FRAME_COUNT)):
        ret, frame = vid.read()
        frameCount = predictions(frame,batCounter,direction)
        finalCount = finalCount + frameCount
        z = z + 1
    else:
        break

print("Final Bat Count for the Entire Video: ", finalCount)

detection_acc = finalCount / total_estimated
print("Detection Accuracy of Model: ", detection_acc)

if (detection_acc > threshold):
    print("The model's accuracy is acceptable")
else:
    print("The model's accuracy is not acceptable")
