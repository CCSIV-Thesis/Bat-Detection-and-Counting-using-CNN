# USAGE
# python video_testing_improved.py --video "Bat Videos/test/video.mp4" --model latestbat.model

import cv2
import numpy as np
import pandas as pd
import argparse
from keras.models import load_model
from keras.utils import to_categorical
from imutils.video import VideoStream
from imutils.video import FPS

#Initializing the bat counter, the output video, and the model
batCounter = 0
ratio = .5  # resize ratio
IMG_SIZE = 64
num_channel = 3
totalBats = 0
total_estimated = 0
framenumber = 0
threshold = 85 #percent: the accuracy of biologist when counting bats
W = None
H = None
bat_ids = []
batidscrossed = []  # blank list to add bat ids that have crossed

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-m", "--model", required=True,
	help="path to trained model")
args = vars(ap.parse_args())

#get vid and model
vid = cv2.VideoCapture(args["video"])
# vid = cv2.VideoCapture('birds.mov')
model = load_model(args["model"])
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("OutputCNNTestDemo3.mp4", fourcc, 30.0, (1920,1080))
frames_count, fps, width, height = vid.get(cv2.CAP_PROP_FRAME_COUNT), vid.get(cv2.CAP_PROP_FPS), int(vid.get(
    cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
df = pd.DataFrame(index=range(int(frames_count)))
#main function
print("Total Estimated Count of Bats: ")
total_estimated = int(input())
print("Which direction does the bats go?\n1 for Right \n2 for Top \n3 for Left \n4 for Down\n")
direction = input()
print("Processing....")

while True:
    ret, frame = vid.read()
    # print("Frame: ",framenumber)
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert image from RGB to GRAY
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,175,50)
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
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,175,50)
        # Apply thresholding on the image to convert to binary image
        thresh_inv = cv2.bitwise_not(thresh)
        # Invert the thresh
        res = cv2.bitwise_and(frame, frame, mask=thresh)
        # Bitwise and with the image mask thresh
        res3 = cv2.bitwise_and(frame, frame, mask=thresh_inv)
        # Bitwise and the image with mask as threshold invert
        image = cv2.addWeighted(res, 1, res3, 1, 0)
        # Take the weighted average
        # final = cv2.drawContours(res4, contours, -1, (0, 255, 0), 1)

		#Outputs from both the canny edge and the watershed algorithm
        # cv2.imshow("Canny Edge Output",canny)
        # cv2.imshow("Watershed Algorithm Output",final)
        # use convex hull to create polygon around contours
        hull = [cv2.convexHull(c) for c in contours]

        # draw contours
        cv2.drawContours(image, hull, -1, (0, 255, 0), 1)

        if(direction == "1"): #Right
            linepos2 = 1740 #1740 #910 #1168
            cv2.line(image, (linepos2, 0), (linepos2, width), (0, 255, 0), 1)
            linepos = 1770 #1770 # 940 #1130
            cv2.line(image, (linepos, 0), (linepos, width), (255, 0, 0), 1)
        elif(direction == "2"): #Top
            linepos2 = 180 #310 / 240
            cv2.line(image, (0, linepos2), (width, linepos2), (0, 255, 0), 1)
            linepos = 150
            cv2.line(image, (0, linepos), (width, linepos), (255, 0, 0), 1)
        elif(direction == "3"): #Left
            linepos = 280
            cv2.line(image, (linepos, 0), (linepos, width), (255, 0, 0), 1)
            linepos2 = 310 #310 / 240
            cv2.line(image, (linepos2, 0), (linepos2, width), (0, 255, 0), 1)
        elif(direction == "4"): #Bottom
            linepos = 440 #570
            cv2.line(image, (0, linepos), (width, linepos), (255, 0, 0), 1)
            linepos2 = 410
            cv2.line(image, (0, linepos2), (width, linepos2), (0, 255, 0), 1)
        # line created to stop counting contours, needed as Bats in distance become one big contour

        # min area for contours in case a bunch of small noise contours are created
        minarea = 10

        # max area for contours, can be quite large for buses
        maxarea = 3000

        # vectors for the x and y locations of contour centroids in current frame
        cxx = np.zeros(len(contours))
        cyy = np.zeros(len(contours))


        # print(image.shape)
        for i in range(len(contours)):  # cycles through all contours in current frame
            if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
                area = cv2.contourArea(contours[i])  # area of contour
                if minarea < area < maxarea:  # area threshold for contour
                    # calculating centroids of contours
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    # cv2.circle(image, (cx,cy), 2, (255,255,255), -1)
                    if(cx > 1 or cy > 1):
                        if(cy-16 < 0 or cx-16 < 0):
                            continue
                        else:
                            if(direction == "1"): #right
                                if cx > linepos2 and cx < linepos+30:  # filters out contours that are above line (y starts at top)
                                    cen_img = image[cy-16:cy+16,cx-16:cx+16]
                                    resized_img = cv2.resize(cen_img,(64,64))
                                    resized_img = resized_img.astype('float32')
                                    resized_img /= 255.0
                                    shape_predict = resized_img.reshape(-1, 64, 64, 3)
                                    prediction = model.predict([shape_predict])
                                    if(prediction[0][0] > prediction[0][1]):
                                        label = "Bat"
                                        prob = prediction[0][0]
                                        label_prob = "{}: {:.2f}%".format(label, prob * 100)
                                        # print(label_prob)
                                        # gets bounding points of contour to create rectangle
                                        # x,y is top left corner and w,h is width and height
                                        x, y, w, h = cv2.boundingRect(cnt)

	                                    # creates a rectangle around contour
                                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

	                                    # Prints centroid text in order to double check later on
                                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
	                                                .3, (0, 0, 255), 1)

                                        cv2.drawMarker(image, (cx, cy), (255, 0, 0), cv2.MARKER_STAR, markerSize=5, thickness=1,
	                                                   line_type=cv2.LINE_AA)

	                                    # adds centroids that passed previous criteria to centroid list
                                        cxx[i] = cx
                                        cyy[i] = cy
                            elif(direction == "3"): #left
                                if cx < linepos2 and cx > linepos-30:  # filters out contours that are above line (y starts at top)
                                    cen_img = image[cy-16:cy+16,cx-16:cx+16]
                                    resized_img = cv2.resize(cen_img,(64,64))
                                    resized_img = resized_img.astype('float32')
                                    resized_img /= 255.0
                                    shape_predict = resized_img.reshape(-1, 64, 64, 3)
                                    prediction = model.predict([shape_predict])
                                    if(prediction[0][0] > prediction[0][1]):
	                                    label = "Bat"
	                                    prob = prediction[0][0]
	                                    label_prob = "{}: {:.2f}%".format(label, prob * 100)
	                                    # print(label_prob)
                                    # gets bounding points of contour to create rectangle
                                    # x,y is top left corner and w,h is width and height
	                                    x, y, w, h = cv2.boundingRect(cnt)

	                                    # creates a rectangle around contour
	                                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

	                                    # Prints centroid text in order to double check later on
	                                    cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
	                                                .3, (0, 0, 255), 1)

	                                    cv2.drawMarker(image, (cx, cy), (255, 0, 0), cv2.MARKER_STAR, markerSize=5, thickness=1,
	                                                   line_type=cv2.LINE_AA)

	                                    # adds centroids that passed previous criteria to centroid list
	                                    cxx[i] = cx
	                                    cyy[i] = cy
                            elif(direction == "2"): #top
                                if cy < linepos2 and cy > linepos-20:  # filters out contours that are above line (y starts at top)
                                    cen_img = image[cy-16:cy+16,cx-16:cx+16]
                                    resized_img = cv2.resize(cen_img,(64,64))
                                    resized_img = resized_img.astype('float32')
                                    resized_img /= 255.0
                                    shape_predict = resized_img.reshape(-1, 64, 64, 3)
                                    prediction = model.predict([shape_predict])
                                    if(prediction[0][0] > prediction[0][1]):
                                        label = "Bat"
                                        prob = prediction[0][0]
                                        label_prob = "{}: {:.2f}%".format(label, prob * 100)
                                        # print(label_prob)
                                        # gets bounding points of contour to create rectangle
                                        # x,y is top left corner and w,h is width and height
                                        x, y, w, h = cv2.boundingRect(cnt)

                                        # creates a rectangle around contour
                                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                                        # Prints centroid text in order to double check later on
                                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                    .3, (0, 0, 255), 1)

                                        cv2.drawMarker(image, (cx, cy), (255, 0, 0), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                                       line_type=cv2.LINE_AA)

                                        # adds centroids that passed previous criteria to centroid list
                                        cxx[i] = cx
                                        cyy[i] = cy
                            elif(direction == "4"): #bottom
                                if cy > linepos2 and cy < linepos+20:  # filters out contours that are above line (y starts at top)
                                    cen_img = image[cy-16:cy+16,cx-16:cx+16]
                                    resized_img = cv2.resize(cen_img,(64,64))
                                    resized_img = resized_img.astype('float32')
                                    resized_img /= 255.0
                                    shape_predict = resized_img.reshape(-1, 64, 64, 3)
                                    prediction = model.predict([shape_predict])
                                    if(prediction[0][0] > prediction[0][1]):
                                        label = "Bat"
                                        prob = prediction[0][0]
                                        label_prob = "{}: {:.2f}%".format(label, prob * 100)
                                        # print(label_prob)
                                        # gets bounding points of contour to create rectangle
                                        # x,y is top left corner and w,h is width and height
                                        x, y, w, h = cv2.boundingRect(cnt)

                                        # creates a rectangle around contour
                                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                                        # Prints centroid text in order to double check later on
                                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                    .3, (0, 0, 255), 1)

                                        cv2.drawMarker(image, (cx, cy), (255, 0, 0), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                                       line_type=cv2.LINE_AA)

                                        # adds centroids that passed previous criteria to centroid list
                                        cxx[i] = cx
                                        cyy[i] = cy

        # eliminates zero entries (centroids that were not added)
        cxx = cxx[cxx != 0]
        cyy = cyy[cyy != 0]

        # empty list to later check which centroid indices were added to dataframe
        minx_index2 = []
        miny_index2 = []

        # maximum allowable radius for current frame centroid to be considered the same centroid from previous frame
        maxrad = 25
        if len(cxx):
           if not bat_ids:
               for i in range(len(cxx)):
                   bat_ids.append(i)
                   df[str(bat_ids[i])] = ""
                   df.at[int(framenumber), str(bat_ids[i])] = [cxx[i], cyy[i]]
                   totalBats = bat_ids[i] + 1
           else:
               # print("Bat ids: ",bat_ids)
               # print(len(cxx))
               # print(len(bat_ids))
               dx = np.zeros((len(cxx),len(bat_ids)))
               dy = np.zeros((len(cxx),len(bat_ids)))
               for i in range(len(cxx)):  # loops through all centroids

                    for j in range(len(bat_ids)):  # loops through all recorded bat ids

                        # acquires centroid from previous frame for specific batid
                        oldcxcy = df.iloc[int(framenumber - 1)][str(bat_ids[j])]
                        # print(oldcxcy)
                        # acquires current frame centroid that doesn't necessarily line up with previous frame centroid
                        curcxcy = np.array([cxx[i], cyy[i]])

                        if not oldcxcy:  # checks if old centroid is empty in case bat leaves screen and new bat shows

                            continue  # continue to next batid

                        else:  # calculate centroid deltas to compare to current frame position later

                            dx[i, j] = oldcxcy[0] - curcxcy[0]
                            dy[i, j] = oldcxcy[1] - curcxcy[1]

               for j in range(len(bat_ids)):  # loops through all current bat ids

                    sumsum = np.abs(dx[:, j]) + np.abs(dy[:, j])  # sums the deltas wrt to bat ids

                    # finds which index batid had the min difference and this is true index
                    correctindextrue = np.argmin(np.abs(sumsum))
                    minx_index = correctindextrue
                    miny_index = correctindextrue

                    # acquires delta values of the minimum deltas in order to check if it is within radius later on
                    mindx = dx[minx_index, j]
                    mindy = dy[miny_index, j]

                    if mindx == 0 and mindy == 0 and np.all(dx[:, j] == 0) and np.all(dy[:, j] == 0):
                        # checks if minimum value is 0 and checks if all deltas are zero since this is empty set
                        # delta could be zero if centroid didn't move

                        continue  # continue to next batid

                    else:

                        # if delta values are less than maximum radius then add that centroid to that specific batid
                        if np.abs(mindx) < maxrad and np.abs(mindy) < maxrad:

                            # adds centroid to corresponding previously existing batid
                            df.at[int(framenumber), str(bat_ids[j])] = [cxx[minx_index], cyy[miny_index]]
                            minx_index2.append(minx_index)  # appends all the indices that were added to previous bat_ids
                            miny_index2.append(miny_index)
               for i in range(len(cxx)):  # loops through all centroids
                    # print(len(cxx))
                    # if centroid is not in the minindex list then another bat needs to be added
                    if i not in minx_index2 and miny_index2:

                        df[str(totalBats)] = ""  # create another column with total Bats
                        totalBats = totalBats + 1  # adds another total bat the count
                        t = totalBats - 1  # t is a placeholder to total Bats
                        bat_ids.append(t)  # append to list of bat ids
                        df.at[int(framenumber), str(t)] = [cxx[i], cyy[i]]  # add centroid to the new bat id

                    elif curcxcy[0] and not oldcxcy and not minx_index2 and not miny_index2:
                        # checks if current centroid exists but previous centroid does not
                        # new bat to be added in case minx_index2 is empty

                        df[str(totalBats)] = ""  # create another column with total Bats
                        totalBats = totalBats + 1  # adds another total bat the count
                        b = totalBats - 1  # t is a placeholder to total Bats
                        bat_ids.append(b)  # append to list of bat ids
                        df.at[int(framenumber), str(b)] = [cxx[i], cyy[i]]  # add centroid to the new bat id

           # print("Bat ids",bat_ids)
           currentBats = 0  # current Bats on screen
           currentBatsindex = []  # current Bats on screen batid index

           for i in range(len(bat_ids)):  # loops through all bat_ids
               if df.at[int(framenumber), str(bat_ids[i])] != '':
                   # checks the current frame to see which bat ids are active
                   # by checking in centroid exists on current frame for certain bat id

                   currentBats = currentBats + 1  # adds another to current Bats on screen
                   currentBatsindex.append(i)  # adds bat ids to current Bats on screen

           for i in range(currentBats):  # loops through all current bat ids on screen
                # grabs centroid of certain batid for current frame
                curcent = df.iloc[int(framenumber)][str(bat_ids[currentBatsindex[i]])]
                # print(curcent)
                # grabs centroid of certain batid for previous frame
                oldcent = df.iloc[int(framenumber - 1)][str(bat_ids[currentBatsindex[i]])]
                # print(oldcent)
                if curcent:  # if there is a current centroid
                    cv2.drawMarker(image, (int(curcent[0]), int(curcent[1])), (0, 0, 255), cv2.MARKER_STAR, markerSize=5,
                               thickness=1, line_type=cv2.LINE_AA)
                    if oldcent:
                        if(direction == "1"):
                            if oldcent[0] <= linepos and curcent[0] >= linepos and bat_ids[currentBatsindex[i]] not in batidscrossed:
                                # print("Counted!")
                                batCounter = batCounter + 1
                                cv2.line(img=image, pt1=(linepos, 0), pt2=(linepos, height), color=(255, 255, 255), thickness=1, lineType=8, shift=0)
                                batidscrossed.append(currentBatsindex[i])  # adds bat id to list of count bats to prevent double counting
                        elif(direction == "4"):
                            if oldcent[1] <= linepos and curcent[1] >= linepos and bat_ids[currentBatsindex[i]] not in batidscrossed:
                                # print("Counted!")
                                batCounter = batCounter + 1
                                cv2.line(img=image, pt1=(0, linepos), pt2=(width, linepos), color=(255, 255, 255), thickness=1, lineType=8, shift=0)
                                batidscrossed.append(currentBatsindex[i])  # adds bat id to list of count bats to prevent double counting
                        elif(direction == "3"):
                            if oldcent[0] >= linepos and curcent[0] <= linepos and bat_ids[currentBatsindex[i]] not in batidscrossed:
                                # print("Counted!")
                                batCounter = batCounter + 1
                                cv2.line(img=image, pt1=(linepos, 0), pt2=(linepos, height), color=(255, 255, 255), thickness=1, lineType=8, shift=0)
                                batidscrossed.append(currentBatsindex[i])  # adds bat id to list of count bats to prevent double counting
                        elif(direction == "2"):
                            if oldcent[1] >= linepos and curcent[1] <= linepos and bat_ids[currentBatsindex[i]] not in batidscrossed:
                                # print("Counted!")
                                batCounter = batCounter + 1
                                cv2.line(img=image, pt1=(0, linepos), pt2=(width, linepos), color=(255, 255, 255), thickness=1, lineType=8, shift=0)
                                batidscrossed.append(currentBatsindex[i])  # adds bat id to list of count bats to prevent double counting
        # Top left hand corner on-screen text
        cv2.rectangle(image, (500, 0), (750, 70), (255, 0, 0), -1)  # background rectangle for on-screen text

        cv2.putText(image, "Bats Counted: " + str(batCounter), (500, 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0),1)

        cv2.putText(image, "Frame: " + str(framenumber) + ' of ' + str(frames_count), (500, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 170, 0), 1)

        cv2.putText(image, 'Time: ' + str(round(framenumber / fps, 2)) + ' sec of ' + str(round(frames_count / fps, 2))
                    + ' sec', (500, 45), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 170, 0), 1)

        cv2.imshow("Bat Counting Process",image)
        out.write(image)
        framenumber = framenumber + 1
        k = cv2.waitKey(int(1000/fps)) & 0xff  # int(1000/fps) is normal speed since waitkey is in ms
        if k == 27:
            break
    else:
        break
#report
print("\n\nEstimated Count of Bats in the video: ", total_estimated)
print("Final Bat Count for the Entire Video: ", batCounter)

#equations in getting percent error, difference between the experimental count and actual count
#as well as the accuracy
temp_count = total_estimated - batCounter
if(total_estimated != 0):
    percent_error = (temp_count / total_estimated) * 100
else:
    percent_error = 0
detection_diff = total_estimated - batCounter
if(total_estimated != 0):
    detection_acc = (batCounter / total_estimated)* 100
else:
    detection_acc = 0

print("Percentage Error of the model: ", percent_error)
print("Detection Performance of the model: ",detection_acc)
print("Difference from Estimate Count of Bats and Detection Count of Bats: ", detection_diff)

if (detection_acc >= threshold and percent_error > -85):
    print("The model's performance is acceptable")
else:
    print("The model's performance is not acceptable")
