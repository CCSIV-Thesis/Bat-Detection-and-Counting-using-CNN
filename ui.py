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

root = Tk()
root.resizable(False,False)
root.geometry('500x500')
root.title("Registration Form")

label_0 = Label(root, text="Bat Counter LOL",width=20,font=("bold", 20))
label_0.place(x=90,y=53)


label_1 = Label(root, text="Video File",width=20,font=("bold", 10))
label_1.place(x=80,y=130)

entry_1 = Entry(root)
entry_1.place(x=240,y=130)

label_2 = Label(root, text="Email",width=20,font=("bold", 10))
label_2.place(x=68,y=180)

entry_2 = Entry(root)
entry_2.place(x=240,y=180)

label_3 = Label(root, text="Gender",width=20,font=("bold", 10))
label_3.place(x=70,y=230)
var = IntVar()
Radiobutton(root, text="Male",padx = 5, variable=var, value=1).place(x=235,y=230)
Radiobutton(root, text="Female",padx = 20, variable=var, value=2).place(x=290,y=230)

label_4 = Label(root, text="country",width=20,font=("bold", 10))
label_4.place(x=70,y=280)

list1 = ['Canada','India','UK','Nepal','Iceland','South Africa'];
c=StringVar()
droplist=OptionMenu(root,c, *list1)
droplist.config(width=15)
c.set('select your country')
droplist.place(x=240,y=280)

label_4 = Label(root, text="Programming",width=20,font=("bold", 10))
label_4.place(x=85,y=330)
var1 = IntVar()
Checkbutton(root, text="java", variable=var1).place(x=235,y=330)
var2 = IntVar()
Checkbutton(root, text="python", variable=var2).place(x=290,y=330)

Button(root, text='Submit',width=20,bg='brown',fg='white').place(x=180,y=380)

root.mainloop()
