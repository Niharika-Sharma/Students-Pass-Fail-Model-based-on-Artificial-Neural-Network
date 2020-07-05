#################################################################################################################
#### GUI Interface for users
#################################################################################################################
from tkinter import *
import tkinter as tk
import tkinter.messagebox

import keras
import numpy as np
from sklearn.preprocessing import StandardScaler

root = Tk()
root.geometry('1500x1500')
root.title("Prediction Form")

label = Label(root, text="Prediction form",width=20,font=("bold", 20))
label.place(x=375,y=20)


label_0 = Label(root, text="Full Name",width=20,font=("bold", 10))
label_0.place(x=0,y=50)
entry_0 = Entry(root)
entry_0.place(x=200,y=50)


label_1 = Label(root, text="Age",width=10,font=("bold", 10))
label_1.place(x=750,y=50)
entry_1 = Entry(root)
entry_1.place(x=950,y=50)


label_2 = Label(root, text="Gender",width=20,font=("bold", 10))
label_2.place(x=0,y=80)
global var2 
var2 = IntVar()
Radiobutton(root, text="Male",padx = 5, variable=var2, value=1).place(x=375,y=80)
Radiobutton(root, text="Female",padx = 20, variable=var2, value=0).place(x=750,y=80)


label_3 = Label(root, text="Address",width=20,font=("bold", 10))
label_3.place(x=0,y=110)
global var3
var3 = IntVar()
Radiobutton(root, text="Urban",padx = 5, variable=var3, value=0).place(x=375,y=110)
Radiobutton(root, text="Rural",padx = 20, variable=var3, value=1).place(x=750,y=110)


label_4 = Label(root, text="Parent's Cohabitation Status",width=24,font=("bold", 10))
label_4.place(x=0,y=140)
global var4
var4 = IntVar()
Radiobutton(root, text="Apart",padx = 5, variable=var4, value=1).place(x=375,y=140)
Radiobutton(root, text="Together",padx = 20, variable=var4, value=0).place(x=750,y=140)


label_5 = Label(root, text="Mother's Education",width=20,font=("bold", 10))
label_5.place(x=0,y=170)
global var5
var5 = IntVar()
Radiobutton(root, text="none",padx = 5, variable=var5, value=0).place(x=250,y=170)
Radiobutton(root, text="primary education",padx = 20, variable=var5, value=1).place(x=400,y=170)
Radiobutton(root, text="5th to 9th grade",padx = 5, variable=var5, value=2).place(x=630,y=170)
Radiobutton(root, text="secondary education",padx = 20, variable=var5, value=3).place(x=820,y=170)
Radiobutton(root, text="higher education",padx = 20, variable=var5, value=4).place(x=1010,y=170)


label_6 = Label(root, text="Father's Education",width=20,font=("bold", 10))
label_6.place(x=0,y=200)
global var6
var6 = IntVar()
Radiobutton(root, text="none",padx = 5, variable=var6, value=0).place(x=250,y=200)
Radiobutton(root, text="primary education",padx = 20, variable=var6, value=1).place(x=400,y=200)
Radiobutton(root, text="5th to 9th grade",padx = 5, variable=var6, value=2).place(x=630,y=200)
Radiobutton(root, text="secondary education",padx = 20, variable=var6, value=3).place(x=820,y=200)
Radiobutton(root, text="higher education",padx = 20, variable=var6, value=4).place(x=1010,y=200)


label_7 = Label(root, text="Mother's Job",width=20,font=("bold", 10))
label_7.place(x=0,y=230)
global var7
var7 = IntVar()
Radiobutton(root, text="teacher",padx = 5, variable=var7, value=4).place(x=250,y=230)
Radiobutton(root, text="health care related",padx = 20, variable=var7, value=1).place(x=400,y=230)
Radiobutton(root, text="services",padx = 5, variable=var7, value=3).place(x=630,y=230)
Radiobutton(root, text="at_home",padx = 20, variable=var7, value=0).place(x=820,y=230)
Radiobutton(root, text="other",padx = 20, variable=var7, value=2).place(x=1010,y=230)



label_8 = Label(root, text="Father's Job",width=20,font=("bold", 10))
label_8.place(x=0,y=260)
global var8
var8 = IntVar()
Radiobutton(root, text="teacher",padx = 5, variable=var8, value=4).place(x=250,y=260)
Radiobutton(root, text="health care related",padx = 20, variable=var8, value=1).place(x=400,y=260)
Radiobutton(root, text="services",padx = 5, variable=var8, value=3).place(x=630,y=260)
Radiobutton(root, text="at_home",padx = 20, variable=var8, value=0).place(x=820,y=260)
Radiobutton(root, text="other",padx = 20, variable=var8, value=2).place(x=1010,y=260)


label_9 = Label(root, text="Travel Time",width=20,font=("bold", 10))
label_9.place(x=0,y=290)
global var9
var9 = IntVar()
Radiobutton(root, text="<15 min",padx = 5, variable=var9, value=1).place(x=270,y=290)
Radiobutton(root, text="15-30 min",padx = 20, variable=var9, value=2).place(x=550,y=290)
Radiobutton(root, text="30-60 min",padx = 5, variable=var9, value=3).place(x=830,y=290)
Radiobutton(root, text=">60 min",padx = 20, variable=var9, value=4).place(x=1110,y=290)


label_10 = Label(root, text="Study Time",width=20,font=("bold", 10))
label_10.place(x=0,y=320)
global var10
var10 = IntVar()
Radiobutton(root, text="<2 hours",padx = 5, variable=var10, value=1).place(x=270,y=320)
Radiobutton(root, text="2 to 5 hours",padx = 20, variable=var10, value=2).place(x=550,y=320)
Radiobutton(root, text="5 to 10 hours",padx = 5, variable=var10, value=3).place(x=830,y=320)
Radiobutton(root, text=">10 hours",padx = 20, variable=var10, value=4).place(x=1110,y=320)


label_11 = Label(root, text="number of past class failures",width=24,font=("bold", 10))
label_11.place(x=0,y=350)
global var11
var11 = IntVar()
Radiobutton(root, text="0",padx = 5, variable=var11, value=0).place(x=270,y=350)
Radiobutton(root, text="1",padx = 20, variable=var11, value=1).place(x=550,y=350)
Radiobutton(root, text="2",padx = 5, variable=var11, value=2).place(x=830,y=350)
Radiobutton(root, text="higher",padx = 20, variable=var11, value=3).place(x=1110,y=350)


label_12 = Label(root, text="Extra Education Support",width=24,font=("bold", 10))
label_12.place(x=0,y=380)
global var12
var12 = IntVar()
Radiobutton(root, text="NO",padx = 5, variable=var12, value=0).place(x=200,y=380)
Radiobutton(root, text="Yes",padx = 20, variable=var12, value=1).place(x=250,y=380)


label_13 = Label(root, text="Extra Paid Classes",width=20,font=("bold", 10))
label_13.place(x=420,y=380)
global var13
var13 = IntVar()
Radiobutton(root, text="NO",padx = 5, variable=var13, value=0).place(x=620,y=380)
Radiobutton(root, text="Yes",padx = 20, variable=var13, value=1).place(x=670,y=380)


label_14 = Label(root, text="Want higher Education",width=20,font=("bold", 10))
label_14.place(x=910,y=380)
global var14
var14 = IntVar()
Radiobutton(root, text="NO",padx = 5, variable=var14, value=0).place(x=1110,y=380)
Radiobutton(root, text="Yes",padx = 20, variable=var14, value=1).place(x=1160,y=380)


label_15 = Label(root, text="Internet Access",width=20,font=("bold", 10))
label_15.place(x=0,y=410)
global var15
var15 = IntVar()
Radiobutton(root, text="NO",padx = 5, variable=var15, value=0).place(x=200,y=410)
Radiobutton(root, text="Yes",padx = 20, variable=var15, value=1).place(x=250,y=410)


label_16 = Label(root, text="Romantic Relationship",width=20,font=("bold", 10))
label_16.place(x=750,y=410)
var16 = IntVar()
Radiobutton(root, text="NO",padx = 5, variable=var16, value=0).place(x=950,y=410)
Radiobutton(root, text="Yes",padx = 20, variable=var16, value=1).place(x=1000,y=410)


label_17 = Label(root, text="Quality of family relationship",width=24,font=("bold", 10))
label_17.place(x=0,y=440)
global var17
var17 = IntVar()
Radiobutton(root, text="1(Very Bad)",padx = 5, variable=var17, value=1).place(x=250,y=440)
Radiobutton(root, text="2",padx = 20, variable=var17, value=2).place(x=410,y=440)
Radiobutton(root, text="3",padx = 5, variable=var17, value=3).place(x=560,y=440)
Radiobutton(root, text="4",padx = 20, variable=var17, value=4).place(x=710,y=440)
Radiobutton(root, text="5(Excellent)",padx = 20, variable=var17, value=5).place(x=860,y=440)


label_18 = Label(root, text="Free time after school",width=24,font=("bold", 10))
label_18.place(x=0,y=470)
global var18
var18 = IntVar()
Radiobutton(root, text="1(Very low)",padx = 5, variable=var18, value=1).place(x=250,y=470)
Radiobutton(root, text="2",padx = 20, variable=var18, value=2).place(x=410,y=470)
Radiobutton(root, text="3",padx = 5, variable=var18, value=3).place(x=560,y=470)
Radiobutton(root, text="4",padx = 20, variable=var18, value=4).place(x=710,y=470)
Radiobutton(root, text="5(Very High)",padx = 20, variable=var18, value=5).place(x=860,y=470)


label_19 = Label(root, text="Going out with friends",width=24,font=("bold", 10))
label_19.place(x=0,y=500)
global var19
var19 = IntVar()
Radiobutton(root, text="1(Very low)",padx = 5, variable=var19, value=1).place(x=250,y=500)
Radiobutton(root, text="2",padx = 20, variable=var19, value=2).place(x=410,y=500)
Radiobutton(root, text="3",padx = 5, variable=var19, value=3).place(x=560,y=500)
Radiobutton(root, text="4",padx = 20, variable=var19, value=4).place(x=710,y=500)
Radiobutton(root, text="5(Very high)",padx = 20, variable=var19, value=5).place(x=860,y=500)


label_20 = Label(root, text="Workday alcohol consumption",width=24,font=("bold", 10))
label_20.place(x=0,y=530)
global var20
var20 = IntVar()
Radiobutton(root, text="1(Very low)",padx = 5, variable=var20, value=1).place(x=250,y=530)
Radiobutton(root, text="2",padx = 20, variable=var20, value=2).place(x=410,y=530)
Radiobutton(root, text="3",padx = 5, variable=var20, value=3).place(x=560,y=530)
Radiobutton(root, text="4",padx = 20, variable=var20, value=4).place(x=710,y=530)
Radiobutton(root, text="5(Very High)",padx = 20, variable=var20, value=5).place(x=860,y=530)


label_21 = Label(root, text="Weekend alcohol consumption",width=24,font=("bold", 10))
label_21.place(x=0,y=560)
global var21
var21 = IntVar()
Radiobutton(root, text="1(Very low)",padx = 5, variable=var21, value=1).place(x=250,y=560)
Radiobutton(root, text="2",padx = 20, variable=var21, value=2).place(x=410,y=560)
Radiobutton(root, text="3",padx = 5, variable=var21, value=3).place(x=560,y=560)
Radiobutton(root, text="4",padx = 20, variable=var21, value=4).place(x=710,y=560)
Radiobutton(root, text="5(Very high)",padx = 20, variable=var21, value=5).place(x=860,y=560)


label_22 = Label(root, text="Current health status",width=24,font=("bold", 10))
label_22.place(x=0,y=590)
global var22
var22 = IntVar()
Radiobutton(root, text="1(Very Bad)",padx = 5, variable=var22, value=1).place(x=250,y=590)
Radiobutton(root, text="2",padx = 20, variable=var22, value=2).place(x=410,y=590)
Radiobutton(root, text="3",padx = 5, variable=var22, value=3).place(x=560,y=590)
Radiobutton(root, text="4",padx = 20, variable=var22, value=4).place(x=710,y=590)
Radiobutton(root, text="5(Very Good)",padx = 20, variable=var22, value=5).place(x=860,y=590)


label_23 = Label(root, text="Absences (Range: 0 to 93) ",width=24,font=("bold", 10))
label_23.place(x=0,y=610)
entry_23 = Entry(root)
entry_23.place(x=375,y=610)



def client_exit():
        root.destroy()

def show_result():
    
        i1 = int(entry_1.get())
        i2 = int(var2.get())
        i3 = int(var3.get())
        i4 = int(var4.get())
        i5 = int(var5.get())
        i6 = int(var6.get())
        i7 = int(var7.get())
        i8 = int(var8.get())
        i9 = int(var9.get())
        i10 = int(var10.get())
        i11 = int(var11.get())
        i12 = int(var12.get())
        i13 = int(var13.get())
        i14 = int(var14.get())
        i15 = int(var15.get())
        i16 = int(var16.get())
        i17 = int(var17.get())
        i18 = int(var18.get())
        i19 = int(var19.get())
        i20 = int(var20.get())
        i21 = int(var21.get())
        i22 = int(var22.get())
        i23 = int(entry_23.get())
        print(i2,i1,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22,i23)
        np.array([[i2,i1,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22,i23]])
        sc = StandardScaler()
        classifier = keras.models.load_model('/home/niharika/Desktop/ML_Project/ANN_student(2).model')

        new_prediction = classifier.predict(sc.fit_transform(np.array([[i2,i1,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22,i23]])))
        a = float(new_prediction)
        print(a)
        if a > 0.5:
            pred = "You will PASS the Examination"
        else:
            pred = "You will FAIL the Examination"

        print(pred)
        tk.messagebox.showinfo( "Prediction", pred )
    
Button(root, text='Submit',width=20,bg='brown',fg='white', command = show_result).place(x=550,y=670)
Button(root, text='EXIT',width=20,bg='brown',fg='white', command = client_exit).place(x=550,y=700)



root.mainloop()


