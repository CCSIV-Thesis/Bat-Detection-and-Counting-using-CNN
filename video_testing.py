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
threshold = 85 #percent: the accuracy of biologist when counting bats

vid = cv2.VideoCapture('output4.mp4')
model = load_model("bat.model")

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
    point = 0
    if(direction == "1" or direction == "3"):
        while(point < y):
            if(direction == "1"):
                crop_img = frame[point:point+32,x:x+32]
            else:
                crop_img = frame[point:point+32,x-32:x]
            # print(crop_img.shape)
            # y_point = 0
            # y64 = crop_img.shape[0]
            # cv2.imshow("cropped",crop_img)
            # # cv2.waitKey(5000)
            # if(y64 == IMG_SIZE):
            #     while(y_point < y64):
            #         x_point = 0
            #         x64 = crop_img.shape[1]
            #         while(x_point < x64-16):
            #             pred_img = crop_img[y_point:y_point+32,x_point:x_point+32]
            #             if(pred_img.shape[0] != x64):
            #                 zeros = np.zeros((32 - pred_img.shape[0])*32*num_channel,dtype="uint8").reshape(((32 - pred_img.shape[0]),32,num_channel))
            #                 pred_img =  np.concatenate((pred_img,zeros))
            #             pred_img = pred_img.astype('float32')
            #             pred_img /= 255.0
            #             # print(pred_img.shape)
            #             cv2.imshow("sliding window",pred_img)
            #             pred_img = cv2.resize(pred_img,(IMG_SIZE,IMG_SIZE))
            #             cv2.imshow("after resize",pred_img)
            #             cv2.waitKey(1)
            #             shape_predict = pred_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            #             prediction = model.predict([shape_predict])
            #             print(prediction)
            #             if(prediction[0][0] > prediction[0][1]):
            #                 print("Bat")
            #                 batCounter = batCounter + 1
            #             else:
            #                 print("Non-Bat")
            #             #50% sliding window
            #             x_point = x_point + 16
            #         y_point = y_point + 16
            if(crop_img.shape[0] != 32):
                zeros = np.zeros((32 - crop_img.shape[0])*32*num_channel,dtype="uint8").reshape(((32 - crop_img.shape[0]),32,num_channel))
                crop_img =  np.concatenate((crop_img,zeros))
            crop_img = crop_img.astype('float32')
            crop_img /= 255.0
            # print(pred_img.shape)
            cv2.imshow("sliding window",crop_img)
            crop_img = cv2.resize(crop_img,(IMG_SIZE,IMG_SIZE))
            cv2.imshow("after resize",crop_img)
            cv2.waitKey(1000)
            shape_predict = crop_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            prediction = model.predict([shape_predict])
            # print(prediction)

            if(prediction[0][0] > prediction[0][1]):
                label = "Bat"
                prob = prediction[0][0]
                label_prob = "{}: {:.2f}%".format(label, prob * 100)
                # print(label)
                batCounter = batCounter + 1
            else:
                label = "Non-Bat"
                prob = prediction[0][1]
                label_prob = "{}: {:.2f}%".format(label, prob * 100)
                # print(label)
            # cv2.putText(crop_img, label, (3, 15),  cv2.FONT_HERSHEY_SIMPLEX,
            #     0.4, (0, 255, 0), 1)
            # crop_img = cv2.resize(crop_img,(512,320))
            # cv2.imshow("Prediction Image",crop_img)
            print(label_prob)
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
            cv2.waitKey(1)
            if(crop_img.shape[0] != IMG_SIZE):
                zeros = np.zeros((IMG_SIZE - crop_img.shape[0])*IMG_SIZE*num_channel,dtype="uint8").reshape(((IMG_SIZE - crop_img.shape[0]),IMG_SIZE,num_channel))
                crop_img =  np.concatenate((crop_img,zeros))
            crop_img = crop_img.astype('float32')
            crop_img /= 255
            shape_predict = crop_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            prediction = model.predict([shape_predict])
            # print(prediction)

            if(prediction[0][0] > prediction[0][1]):
                label = "Bat"
                prob = prediction[0][0]
                label_prob = "{}: {:.2f}%".format(label, prob * 100)
                # print(label)
                batCounter = batCounter + 1
            else:
                label = "Non-Bat"
                prob = prediction[0][1]
                label_prob = "{}: {:.2f}%".format(label, prob * 100)
                # print(label)
            # cv2.putText(crop_img, label, (3, 15),  cv2.FONT_HERSHEY_SIMPLEX,
            #     0.4, (0, 255, 0), 1)
            # crop_img = cv2.resize(crop_img,(512,320))
            # cv2.imshow("Prediction Image",crop_img)
            print(label_prob)

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

#report
print("\n\nEstimated Count of Bats in the video: ", total_estimated)
print("Final Bat Count for the Entire Video: ", finalCount)

#equations in getting percent error, difference between the experimental count and actual count
#as well as the accuracy
percent_error = ((total_estimated - finalCount) / total_estimated) * 100
detection_diff = total_estimated - finalCount
detection_acc = (finalCount / total_estimated)* 100

print("Percentage Error of the model: ", percent_error)
print("Detection Performance of the model: ",detection_acc)
print("Difference from Estimate Count of Bats and Detection Count of Bats: ", detection_diff)

if (detection_acc >= threshold and detection_acc < 100):
    print("The model's performance is acceptable")
else:
    print("The model's performance is not acceptable")
