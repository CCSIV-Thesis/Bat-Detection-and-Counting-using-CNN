# import tkinter as tk
# from tkinter import ttk
# from tkinter import filedialog
# from tkinter import messagebox
# from tkinter.ttk import *
#
# class SampleApp(tk.Tk):
#
#     def __init__(self, *args, **kwargs):
#         tk.Tk.__init__(self, *args, **kwargs)
#         self.minsize(1280,720)
#         self.button = ttk.Button(text="start", command=self.start)
#         self.button.pack()
#         self.button1 = ttk.Button(text="select file", command=self.getFile)
#         self.button1.pack()
#         self.combo = Combobox(self)
#         self.combo['values']= ("Left", "Right", "Top", "Bottom")
#         self.combo.current(0) #set the selected item
#         self.combo.pack()
#         self.progress = ttk.Progressbar(self, orient="horizontal",
#                                         length=200, mode="determinate")
#
#         self.progress.pack()
#         self.bytes = 0
#         self.maxbytes = 0
#
#     def getFile(self):
#         self.file = filedialog.askopenfilename(filetypes = (("Video files","*.mp4"),("all files","*.*")))
#         # self.dir = filedialog.askdirectory()
#
#         print(self.file)
#     def start(self):
#         self.progress["value"] = 0
#         self.maxbytes = 10000
#         self.progress["maximum"] = 10000
#         self.read_bytes()
#
#     def read_bytes(self):
#         '''simulate reading 500 bytes; update progress bar'''
#         self.bytes += 500
#         self.progress["value"] = self.bytes
#         if self.bytes == self.maxbytes:
#             messagebox.showinfo('Done!', 'Done')
#         elif self.bytes < self.maxbytes:
#             # read more bytes after 100 ms
#             self.after(100, self.read_bytes)
#
#
# app = SampleApp()
# app.mainloop()


from tkinter import *
import cv2
import numpy as np
from keras.models import load_model
try:
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter import filedialog
    from tkinter import messagebox
except ImportError:
    import Tkinter as tk
    import ttk
    import tkFileDialog as filedialog
from tkfilebrowser import askopendirname, askopenfilenames, asksaveasfilename
from PIL import ImageTk, Image


model = load_model("latestbat.model")

root = Tk()
root.resizable(False,False)
root.geometry('1000x700')
root.title("BAT DETECTION AND COUNTING USING CNN")
def clicked():
    batCounter = 0
    finalCount = 0
    z = 0
    vid_path = vid_lbl.cget("text")
    if(vid_path == "Select Video"):
        messagebox.showinfo('Must choose file!', 'You must choose a video file!')
    else:
        startCounting(vid_path,batCounter,finalCount,z)
            # ret, frame = vid.read()
            # main_frame.image=frame
            # cv2.imshow("Video!",frame)
            # cv2.waitKey(1000)
            # if cv2.waitKey(1) & 0xFF == ord('q'):  # if 'q' is pressed then quit
            #     break
        # vid.release()
        # cv2.destroyAllWindows()

def startCounting(vid_path,batCounter,finalCount,z):
    vid = cv2.VideoCapture(vid_path)
    while vid.isOpened():
        if(z < vid.get(cv2.CAP_PROP_FRAME_COUNT)):
            ret, frame = vid.read()
            main_frame.image=frame
            preprocessedFrame = preprocessing(frame)
            frameCount = predictions(preprocessedFrame,batCounter)
            finalCount = finalCount + frameCount
            bCounterText.configure(text=finalCount)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # if 'q' is pressed then quit
                break
            z = z + 1
            batCounter = 0
            root.after(1000,startCounting(vid_path,batCounter,finalCount,z))
        else:
            break
def c_open_file_old():
    vid_open = filedialog.askopenfilenames(parent=root, initialdir='/Users/KuyaKoya/Documents/Bat-Detection-and-Counting-using-CNN/', initialfile='',
                                      filetypes=[("mp4", "*.mp4")])
    vid_lbl.configure(text=vid_open[0])
    print(vid_open[0])

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

def predictions(frame,batCounter):
    direction = droplist.cget("text")
    # print(direction)
    IMG_SIZE = 64
    num_channel = 3
    if(direction == "RIGHT"): #Right
        x = frame.shape[1] - 150
        y = frame.shape[0]
    elif(direction == "TOP"): #Top
        x = frame.shape[1]
        y = frame.shape[0] - 570 #276
    elif(direction == "LEFT"): #Left
        x = frame.shape[1] - 1130 #490
        y = frame.shape[0]
    elif(direction == "BOTTOM"): #Bottom
        x = frame.shape[1]
        y = frame.shape[0] - 150
    point = 0
    if(direction == "RIGHT" or direction == "LEFT"):
        # print("X: ",x)
        while(point < y):
            if(direction == "RIGHT"):
                crop_img = frame[point:point+IMG_SIZE,x:x+IMG_SIZE]
            else:
                crop_img = frame[point:point+IMG_SIZE,x-IMG_SIZE:x]
            crop_frame.image=crop_img
            if(crop_img.shape[0] != IMG_SIZE):
                zeros = np.zeros((IMG_SIZE - crop_img.shape[0])*IMG_SIZE*num_channel,dtype="uint8").reshape(((IMG_SIZE - crop_img.shape[0]),IMG_SIZE,num_channel))
                crop_img =  np.concatenate((crop_img,zeros))
            crop_img = crop_img.astype('float32')
            crop_img /= 255
            shape_predict = crop_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            prediction = model.predict([shape_predict])
            if(prediction[0][0] > prediction[0][1]):
                batCounter = batCounter + 1
            point = point + 64
    elif(direction == "TOP" or direction == "BOTTOM"):
        while(point < x):
            if(direction == "TOP"):
                crop_img = frame[y-IMG_SIZE:y,point:point+IMG_SIZE]
            else:
                crop_img = frame[y:y+IMG_SIZE,point:point+IMG_SIZE]
            if(crop_img.shape[0] != IMG_SIZE):
                zeros = np.zeros((IMG_SIZE - crop_img.shape[0])*IMG_SIZE*num_channel,dtype="uint8").reshape(((IMG_SIZE - crop_img.shape[0]),IMG_SIZE,num_channel))
                crop_img =  np.concatenate((crop_img,zeros))
            crop_img = crop_img.astype('float32')
            crop_img /= 255
            shape_predict = crop_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            prediction = model.predict([shape_predict])
            # print(shape_predict)
            # print(prediction)
            if(prediction[0][0] > prediction[0][1]):
                batCounter = batCounter + 1
            point = point + 64
    # print("Bat Count for this Frame: ", batCounter)
    return batCounter


label_0 = Label(root, text="Bat Counter",width=20,font=("bold", 30))
label_0.place(x=310,y=10)


video = Label(root, text="Video File",width=20,font=("bold", 20))
video.place(x=75,y=70)

vid_lbl = Label(root, text="Select Video")
vid_lbl.config(state = NORMAL)
# vid_lbl.config(state="readonly")
vid_lbl.place(x=300, y=70)

vid_open = Button(root, text="Open file", command=c_open_file_old)
vid_open.place(x=300,y=100)

label_4 = Label(root, text="Direction",width=20,font=("bold", 20))
label_4.place(x=75,y=120)

list1 = ['LEFT','RIGHT','BOTTOM','TOP'];
c=StringVar()
droplist=OptionMenu(root,c, *list1)
droplist.config(width=20)
c.set('Select Direction')
droplist.place(x=300,y=125)

submit = Button(root, text='Submit',width=20,bg='white',fg='green',command=clicked)
submit.place(x=410,y=180)

progress = ttk.Progressbar(root, orient="horizontal",
                                length=200, mode="determinate").place(x=400,y=210)
main_frame = "batmanlogo.jpg"
crop_frame = "sunset.jpg"
classification = "Bat"
batCountText = "Bat Count:"
batCountInt = 0

#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
#this is for the main frame
main_frame = ImageTk.PhotoImage(Image.open(main_frame))
panel = tk.Label(root, image = main_frame)
panel.place(x=150,y=450)

#this is for the crop frame
crop_frame = ImageTk.PhotoImage(Image.open(crop_frame).resize((64,64), Image.ANTIALIAS))
panel = tk.Label(root, image = crop_frame)
panel.place(x=550,y=450)

#this is for the classification
label = Label(root, text=classification,width=10,font=("bold", 14)).place(x=560, y=550)

#this is for the current bat count
bCounter = Label(root, text=batCountText,width=12,font=("bold", 14)).place(x=410, y=240)
bCounterText = Label(root, text=batCountInt,width=5,font=("bold", 14))
bCounterText.place(x=550, y=240)


root.mainloop()
