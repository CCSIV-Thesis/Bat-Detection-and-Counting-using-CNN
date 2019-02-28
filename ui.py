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

try:
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter import filedialog
except ImportError:
    import Tkinter as tk
    import ttk
    import tkFileDialog as filedialog
from tkfilebrowser import askopendirname, askopenfilenames, asksaveasfilename
from PIL import ImageTk, Image


root = Tk()
root.resizable(False,False)
root.geometry('1000x700')
root.title("BAT DETECTION AND COUNTING USING CNN")

class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.minsize(1280,720)
        self.button = ttk.Button(text="start", command=self.start)
        self.button.pack()
        self.button1 = ttk.Button(text="select file", command=self.getFile)
        self.button1.pack()
        self.combo = Combobox(self)
        self.combo['values']= ("Left", "Right", "Top", "Bottom")
        self.combo.current(0) #set the selected item
        self.combo.pack()
        self.progress = ttk.Progressbar(self, orient="horizontal",
                                        length=200, mode="determinate")

        self.progress.pack()
        self.bytes = 0
        self.maxbytes = 0

    def getFile(self):
        self.file = filedialog.askopenfilename(filetypes = (("Video files","*.mp4"),("all files","*.*")))
        # self.dir = filedialog.askdirectory()

        print(self.file)
    def start(self):
        self.progress["value"] = 0
        self.maxbytes = 10000
        self.progress["maximum"] = 10000
        self.read_bytes()

    def read_bytes(self):
        '''simulate reading 500 bytes; update progress bar'''
        self.bytes += 500
        self.progress["value"] = self.bytes
        if self.bytes == self.maxbytes:
            messagebox.showinfo('Done!', 'Done')
        elif self.bytes < self.maxbytes:
            # read more bytes after 100 ms
            self.after(100, self.read_bytes)

def clicked():
    root = SampleApp()

def c_open_file_old():
    vid_open = filedialog.askopenfilenames(parent=root, initialdir='/Users/KuyaKoya/Documents/Bat-Detection-and-Counting-using-CNN/', initialfile='',
                                      filetypes=[("mp4", "*.mp4")])

    vid_lbl.configure(text=vid_open)
    print(vid_open)


label_0 = Label(root, text="Bat Counter",width=20,font=("bold", 30))
label_0.place(x=310,y=10)


video = Label(root, text="Video File",width=20,font=("bold", 20))
video.place(x=75,y=70)

vid_lbl = Label(root, text="select video")
vid_lbl.config(state = NORMAL)
# vid_lbl.config(state="readonly")
vid_lbl.place(x=300, y=70)

vid_open = Button(root, text="Open files", command=c_open_file_old)
vid_open.place(x=300,y=100)

label_4 = Label(root, text="Direction",width=20,font=("bold", 20))
label_4.place(x=75,y=120)

list1 = ['LEFT','RIGHT','DOWN','UP'];
c=StringVar()
droplist=OptionMenu(root,c, *list1)
droplist.config(width=20)
c.set('Select Direction')
droplist.place(x=300,y=125)

submit = Button(root, text='Submit',width=20,bg='green',fg='green', command=clicked).place(x=410,y=180)


main_frame = "batmanlogo.jpg"
crop_frame = "b.png"
classification = "Bat"

#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
#this is for the main frame
main_frame = ImageTk.PhotoImage(Image.open(main_frame))
panel = tk.Label(root, image = main_frame)
panel.place(x=150,y=450)

#this is for the crop frame
crop_frame = ImageTk.PhotoImage(Image.open(crop_frame))
panel = tk.Label(root, image = crop_frame)
panel.place(x=550,y=450)

#this is for the classification
label = Label(root, text=classification,width=10,font=("bold", 20)).place(x=560, y=550)

root.mainloop()
