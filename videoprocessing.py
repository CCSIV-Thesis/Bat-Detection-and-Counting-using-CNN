import cv2
import numpy as np
from keras.models import load_model

IMG_SIZE = 128
def preprocess_video(outPath,fps,mirror=False):
    # Capturing video from webcam:
    cap = cv2.VideoCapture('Bat+Flyout+2+7_22_17.mp4')
    #Bat+Flyout+2+7_22_17.mp4 / batsflyout.mp4
    currentFrame = 0

    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(outPath, fourcc, fps, (int(width), int(height)))

    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            if mirror == False:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Convert image from RGB to GRAY
                ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                # apply thresholding to convert the image to binary
                fg = cv2.erode(thresh, None, iterations=1)
                # erode the image
                bgt = cv2.dilate(thresh, None, iterations=1)
                # Dilate the image
                ret, bg = cv2.threshold(bgt, 1, 255, 1)
                # Apply thresholding
                marker = cv2.add(fg, bg)
                # Add foreground and background
                canny = cv2.Canny(marker, 125, 150)
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
            # Saves for video
            # out.write(frame)
            out.write(final)
            cv2.namedWindow("Canny+Watershed", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Canny+Watershed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Canny+Watershed", final)  # Display the image
            cv2.imshow("Canny", canny)  # Display the image
            cv2.imshow("Grayscale", gray)  # Display the image
            cv2.imshow("Foreground", fg)  # Display the image
            cv2.imshow("Background", bg)  # Display the image
            cv2.imshow("Marker", marker)  # Display the image
            cv2.imshow("Thresh", res)  # Display the image
            cv2.imshow("Thresh Inverse", res3)  # Display the image

            # Display the resulting frame
            cv2.imshow('frame', frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):  # if 'q' is pressed then quit
            break

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    preprocess_video('output.mp4', 30.0,mirror=False)

if __name__ == '__main__':
    main()
