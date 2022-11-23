from tkinter import *
import EmotionDetection.src.emotions as emo
import ObjectDetection.yolov5.detect as det
import os

window = Tk()
window.title("Welcome to VniTeach app")
window.geometry('350x200')
lbl = Label(window, text="Hello")
lbl.grid(column=0, row=0)

#Hàm khi nút được nhấn
def clicked():
    lbl.configure(text="Button was clicked !!")
    emo.emotion()

#Gọi hàm clicked khi nút được nhấn
btn = Button(window, text="Click Me", command=clicked)
btn.grid(column=1, row=0)

window.mainloop()