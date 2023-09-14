import cv2
import cvzone
import face_recognition
from picamera2 import Picamera2
import numpy as np
import mysql.connector
from mysql.connector import Error
import tensorflow as tf
import tensorflow.lite as lite
import os
from datetime import datetime,date,time,timedelta
import yagmail
import pygame
import RPi.GPIO as GPIO
import pandas as pd
from joblib import load

#-------------------------------MySQL Commads----------------------------
host_name, user_name, password, database= ("localhost" ,"root", "12345908527", "Students")

def my_connection(host_name, user_name, password, database):
        connection = None 
        try:
            connection = mysql.connector.connect(
            host = host_name,
            user = user_name,
            passwd = password,
            db = database
            )
            print("Successful")
        except Error as err:
            print(f"Error:{err}")
        return connection

def read_query(connection, query):
        cursor = connection.cursor()
        result = None
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            return results
        except Error as err:
            print(f"Error : {err}")
def execute_query(connection, query):
        cursor = connection.cursor()
        try:
            cursor.execute(query)
            connection.commit()
            print("Successfully Executed")
        except Error as err:
            print(f"Error f{err}")
connection = my_connection(host_name, user_name, password, database)


#----------------------------------------------------------------------------------
#-------------------------------Anti-Spoofing----------------------------------
class ClassifierLite:
    def __init__(self, model, labels):
        self.interpreter = tf.lite.Interpreter(model_path=model)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        with open(labels, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]
            self.true_labels = [label.split(" ")[1] for label in self.labels]
        
    def get_prediction(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.asarray(img, dtype=np.uint8)
        self.input_data = np.expand_dims(img, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], self.input_data)
        self.interpreter.invoke()
        self.output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        self.predicted_class = self.true_labels[np.argmax(self.output_data)]
        return  self.predicted_class
#----------------------------------------------------------------------------------
#-------------------------------Play Audio--------------------------------------
class playAudio:
    def __init__(self, file):
        self.sound = file
        pygame.mixer.init()
        sound_file = pygame.mixer.Sound(self.sound)
        sound_file.play()
#----------------------------------------------------------------------------------
#-------------------------------TIME MODE-----------------------------------
time_in_picture = cv2.imread("interface2/TIME IN.png")
time_out_picture = cv2.imread("interface2/TIME OUT.png")
cutting_picture = cv2.imread("interface2/Cutting.png")
GPIO.setmode(GPIO.BCM)
time_in_pin =  17
time_out_pin = 27
GPIO.setup(time_in_pin, GPIO.IN)
GPIO.setup(time_out_pin, GPIO.IN)
value_state1 = 0
value_state2 = 1
time_in_option = 0
time_out_option = 0
#----------------------------------------------------------------------------------
def email_sender_parents(email, lastname, firstname, image):
    yag = yagmail.SMTP("calambasti5@gmail.com", "oeyoxzeqgwqvuyyv")
    yag.send(to =email, subject="Student Attendace", contents= f"""<h1>{lastname}, {firstname} is now inside the school.</h1>
Kindly Check if the person in the image is your son/daugther.
Have a good day.
Aim High With STI.
""", attachments=image)
    print("Email Sent")
def email_sender_security(email, lastname, firstname, image):
    yag = yagmail.SMTP("calambasti5@gmail.com", "oeyoxzeqgwqvuyyv")
    yag.send(to =email, subject="Student Attendace", contents=f"""<h1>{lastname} {firstname} an anonymous person is trying to enter the school.</h1>
Please verify if the person in the picture is allowed to enter the school.
Have a good day.
Aim High With STI.
""", attachments=image)
    print("Email Sent")
    
#-------------------------------CAMERA SETTINGS------------------------------
piCam = Picamera2()
piCam.preview_configuration.main.size=(720, 600)
piCam.preview_configuration.main.format="RGB888"
piCam.preview_configuration.controls.FrameRate = 60
piCam.configure("preview")
piCam.start()
#-----------------------------------------------------------------------------------------

#-------------------------------ALL OF THE CLASSES---------------------------
myModel = ClassifierLite(model="models/model.tflite", labels="models/labels.txt")
#-----------------------------------------------------------------------------------------

#----------------------INTERFACE OF THE FINAL PROJECT-----------------
path2 = "interface"
guiList = os.listdir(path2)
guiInterFace = []
for img in guiList:
    gui = cv2.imread(f"{path2}/{img}")
    guiInterFace.append(gui)


webcam = guiInterFace[2]
#-----------------------------------------------------------------------------------------
    
#----------------------ENCODING THE TRAIN DATA---------------------------
path = "train_data"
images_path ="images"
imageList = os.listdir(path)
images_gui = os.listdir(images_path)
images_text = []
images = []
img_for_gui = []
id_numbers = []

for img in images_gui:
    curImg = cv2.imread(f"{path}/{img}")
    curImg2 = cv2.resize(curImg, (245, 215))
    img_for_gui.append(curImg2)
    image_num = os.path.splitext(img)[0]
    images_text.append(int(image_num))

    
for img in imageList:
    id = os.path.splitext(img)[0]
    real_id = id.split(" ")[0]
    id_numbers.append(int(real_id))


encodeKnown = load(filename = "encodeListKnown.joblib")
#-----------------------------------------------------------------------------------------

#---------------------------------------TIME-------------------------------------------
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
result_time = datetime.combine(date.today(),time(int(current_time[:2]),
                                            int(current_time[3:5]),
                                            int(current_time[6:8]))) + timedelta(seconds=5)
now2 = datetime.now()
current_time2 = now2.strftime("%H:%M:%S")
result_time2 = datetime.combine(date.today(),time(int(current_time2[:2]),
                                            int(current_time2[3:5]),
                                            int(current_time2[6:8]))) + timedelta(seconds=5)
now3 = datetime.now()
current_time3 = now3.strftime("%H:%M:%S")
result_time3 = datetime.combine(date.today(),time(int(current_time3[:2]),
                                            int(current_time3[3:5]),
                                            int(current_time3[6:8]))) + timedelta(seconds=5)

now4 = datetime.now()
current_time4 = now4.strftime("%H:%M:%S")
result_time4 = datetime.combine(date.today(),time(int(current_time4[:2]),
                                            int(current_time4[3:5]),
                                            int(current_time4[6:8]))) + timedelta(seconds=5)
#----------------------------------------------------------------------------------------
intruder_attack = 0

#----------------------------------CSV DATA----------------------------------------
Student_ids = []
names_of_students = []
time_of_entries = []
time_of_out = []
data = None
string_date_now = now.strftime("%Y-%m-%d")


#----------------------------------------------------------------------------------------
while True:
    curTime = datetime.now()
    curTime_string = curTime.strftime("%H:%M:%S")
    if curTime_string =="20:12:00":
        result_for_attendees = read_query(connection, f"""SELECT Student_ID, CONCAT(Last_Name,",", "", First_Name), Time_In, Time_Out
FROM StudentAttendance LEFT JOIN Students USING(ID) """)
        for id in result_for_attendees:
            Student_ids.append(id[0])
            names_of_students.append(id[1])
            time_of_entries.append(id[2])
            time_of_out.append(id[3])
        print(Student_ids, names_of_students, time_of_entries, time_of_out)
        data = pd.DataFrame({
        "Name of Students": names_of_students,
        "Time of entry": time_of_entries,
        "Time of out": time_of_out
            }, index= Student_ids)
        data.to_csv(f"attendance/{string_date_now}.csv")
        execute_query(connection, "TRUNCATE TABLE StudentAttendance")
    value_state1 = GPIO.input(time_in_pin)
    value_state2 = GPIO.input(time_out_pin)   
    
    img = piCam.capture_array()
    image = img.copy()
    class_name = myModel.get_prediction(img)
    webcam[0:600, 0:720] = image
    webcam[0:600, 600: 424 + 600] = guiInterFace[0]
    if value_state1 == 1:
        time_in_option = 1
        time_out_option =0
    elif value_state2 == 1:
        time_in_option = 0
        time_out_option = 1

    
    if time_in_option == 1:
        webcam[0:107, 0:210] = time_in_picture
    elif time_in_option == 0:
        webcam[0:107, 0:210] = time_out_picture
    
    if class_name == "Face":
        imgS = cv2.resize(image, (0,0), None, 0.25, 0.25)
        imgRgb = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        curFaceLoc = face_recognition.face_locations(imgRgb)
        curEncoding = face_recognition.face_encodings(imgRgb, curFaceLoc,num_jitters=0,model="small")
        for encodFace, faceLoc in zip(curEncoding, curFaceLoc):
            matches  = face_recognition.compare_faces(encodeKnown, encodFace, tolerance= 0.40)
            faceDis = face_recognition.face_distance(encodeKnown, encodFace)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1  = y1 *4, x2 * 4, y2*4,x1*4
            time_in = read_query(connection, "SELECT * FROM StudentAttendance ")
            inside = [id[0] for id in time_in]
            bbox = x1, y1, x2-x1, y2-y1
                    
            if time_in_option== 1 and time_out_option == 0:
                try:
                    _ = matches.index(True)
                    num_index = np.argmin(faceDis)
                    student_id = id_numbers[num_index]
                    StudentsInfo = read_query(connection, f"SELECT * FROM Students WHERE ID = {student_id}" )[0]
                    _, Student_Id, LastName, FirstName, Program, Attendance_Count, Contanct_Number, Email = StudentsInfo
                
                    if student_id not in inside:
                        playAudio("sounds/Face_Recognized_voice.wav")
                        enter_attendance = f"""INSERT INTO StudentAttendance(ID) VALUES({student_id})"""
                        execute_query(connection, enter_attendance)
                        intruder_attack = 0
                        try:
                            img_path = f"entry_picture/{student_id}_in_img.jpg"
                            cv2.imwrite(img_path, image)
#                             email_sender_parents(Email, LastName, FirstName, img_path)
#                             os.remove(img_path)
                        except:
                            print("No internet")
                   
                    else:
                        webcam[0:600, 600:424 + 600]= guiInterFace[5]
                        student_img = images_text.index(student_id)
                        webcam[135:215 + 135, 688:245 + 688] = img_for_gui[student_img]
                        cv2.putText(webcam,f"{LastName}",(680, 412), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (255,255,255), 2)
                        cv2.putText(webcam,f"{Student_Id}",(680, 485), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6, (255,255,255), 2)
                        cv2.putText(webcam,f"{Program}",(680, 560), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6, (255,255,255), 2)
                        intruder_attack = 0
                    
                except:
                    cvzone.cornerRect(webcam, bbox, rt=0, colorC=(255, 0, 0))
                    webcam[0:600, 601:423 + 601] = guiInterFace[3]
                    if curTime > result_time:
                        playAudio("sounds/Warning_voice.wav")
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        result_time = datetime.combine(date.today(),time(int(current_time[:2]),
                                                                int(current_time[3:5]),
                                                                int(current_time[6:8]))) + timedelta(seconds=5)
                        intruder_attack += 1
                        if intruder_attack == 5:
                            intrud_img = "intrud_img.jpg"
                            cv2.imwrite(intrud_img, image)
                            email_sender_security("moradacarl2711@gmail.com", "High Officer", "Security", intrud_img)
                            os.remove(intrud_img)
                            intruder_attack == 0          
                    
                else:
                    cvzone.cornerRect(webcam, bbox, rt=0)
                    cvzone.putTextRect(webcam,f"{FirstName}", (x2 -130, y2  + 70), thickness=3, scale = 2)
                    
            elif time_in_option == 0 and time_out_option == 1:
                try:
                    _ = matches.index(True)
                    record_curTime = datetime.now()
                    num_index = np.argmin(faceDis)
                    student_id = id_numbers[num_index]
                    StudentsInfo = read_query(connection, f"""SELECT * FROM Students
                                                                                                                    LEFT JOIN StudentAttendance
                                                                                                                    USING(ID) WHERE ID = {student_id}""" )[0]
                    _, Student_Id, LastName, FirstName, Program, Attendance_Count, Contanct_Number, Email, Time_In, Time_Out,time_in_emai,time_out_email= StudentsInfo
                    if Time_Out == None:
                        if _ not in inside and curTime > result_time3:
                            playAudio("sounds/Not in.wav")
                            now3 = datetime.now()
                            current_time3 = now3.strftime("%H:%M:%S")
                            result_time3 = datetime.combine(date.today(),time(int(current_time2[:2]),
                                                        int(current_time3[3:5]),
                                                        int(current_time3[6:8]))) + timedelta(seconds=5)
                            
                        elif _ in inside:
                            my_now = datetime.now()
                            day_of_week = my_now.weekday()
                            if day_of_week == 0:
                                monday_query = f"SELECT * FROM Schedule WHERE ID = {student_id}"
                                my_result_date = str(read_query(connection, monday_query)[0][1])
                                print(my_result_date)
                                my_result_date = time(hour=int(my_result_date[:2]), minute =int(my_result_date[3:5]), second =int(my_result_date[6:]))
                                time_now = now.time()
                                if time_now > my_result_date:
                                    intruder_attack = 0
                                    playAudio("sounds/Face_Recognized_voice.wav")
                                    time_out_now = record_curTime.strftime("%Y-%m-%d %H:%M:%S")
                                    execute_query(connection, f'UPDATE StudentAttendance SET Time_Out = "{time_out_now}" WHERE Id = {student_id}')
                                    img_path = f"entry_picture/{student_id}_out_img.jpg"
                                    cv2.imwrite(img_path, image)
                                    if Time_In != None:
                                        execute_query(connection, f"""UPDATE Students
                                                                                                 SET Attendance_Count = {Attendance_Count + 1}
                                                                                                 WHERE Id = {student_id}""")
                                elif time_now < my_result_date:
                                    intruder_attack = 0
                                    webcam[0:600, 600:424 + 600] = cutting_picture
                                    if curTime > result_time4:
                                        playAudio("sounds/Cutting.wav")
                                        now4 = datetime.now()
                                        current_time4 = now4.strftime("%H:%M:%S")
                                        result_time4 = datetime.combine(date.today(),time(int(current_time4[:2]),
                                                                    int(current_time4[3:5]),
                                                                    int(current_time4[6:8]))) + timedelta(seconds=5)
                        
                                
                            elif day_of_week == 1:
                                tuesday_query = f"SELECT * FROM Schedule WHERE ID = {student_id}"
                                my_result_date = str(read_query(connection, tuesday_query)[0][2])
                                my_result_date = time(hour=int(my_result_date[:2]), minute =int(my_result_date[3:5]), second =int(my_result_date[6:]))
                                time_now = now.time()
                                if time_now > my_result_date:
                                    playAudio("sounds/Face_Recognized_voice.wav")
                                    time_out_now = record_curTime.strftime("%Y-%m-%d %H:%M:%S")
                                    execute_query(connection, f'UPDATE StudentAttendance SET Time_Out = "{time_out_now}" WHERE Id = {student_id}')
                                    img_path = f"entry_picture/{student_id}_out_img.jpg"
                                    cv2.imwrite(img_path, image)
                                    if Time_In != None:
                                        execute_query(connection, f"""UPDATE Students
                                                                                                 SET Attendance_Count = {Attendance_Count + 1}
                                                                                                 WHERE Id = {student_id}""")
                                else:
                                    intruder_attack = 0
                                    webcam[0:600, 600:424 + 600] = cutting_picture
                                    if curTime > result_time4:
                                        playAudio("sounds/Cutting.wav")
                                        now4 = datetime.now()
                                        current_time4 = now4.strftime("%H:%M:%S")
                                        result_time4 = datetime.combine(date.today(),time(int(current_time4[:2]),
                                                                    int(current_time4[3:5]),
                                                                    int(current_time4[6:8]))) + timedelta(seconds=5)
                                    
                            elif day_of_week == 2:
                                wed_query = f"SELECT * FROM Schedule WHERE ID = {student_id}"
                                my_result_date = str(read_query(connection, wed_query)[0][3])
                                my_result_date = time(hour=int(my_result_date[:2]), minute =int(my_result_date[3:5]), second =int(my_result_date[6:]))
                                time_now = now.time()
                                if time_now > my_result_date:
                                    playAudio("sounds/Face_Recognized_voice.wav")
                                    time_out_now = record_curTime.strftime("%Y-%m-%d %H:%M:%S")
                                    execute_query(connection, f'UPDATE StudentAttendance SET Time_Out = "{time_out_now}" WHERE Id = {student_id}')
                                    img_path = f"entry_picture/{student_id}_out_img.jpg"
                                    cv2.imwrite(img_path, image)
                                    if Time_In != None:
                                        execute_query(connection, f"""UPDATE Students
                                                                                                 SET Attendance_Count = {Attendance_Count + 1}
                                                                                                 WHERE Id = {student_id}""")
                                
                                else:
                                    intruder_attack = 0
                                    webcam[0:600, 600:424 + 600] = cutting_picture
                                    if curTime > result_time4:
                                        playAudio("sounds/Cutting.wav")
                                        now4 = datetime.now()
                                        current_time4 = now4.strftime("%H:%M:%S")
                                        result_time4 = datetime.combine(date.today(),time(int(current_time4[:2]),
                                                                    int(current_time4[3:5]),
                                                                    int(current_time4[6:8]))) + timedelta(seconds=5)
                     
                            elif day_of_week == 3:
                                thurs_query = f"SELECT * FROM Schedule WHERE ID = {student_id}"
                                my_result_date = str(read_query(connection, thurs_query)[0][4])
                                my_result_date = time(hour=int(my_result_date[:2]), minute =int(my_result_date[3:5]), second =int(my_result_date[6:]))
                                time_now = now.time()
                                if time_now > my_result_date:
                                    playAudio("sounds/Face_Recognized_voice.wav")
                                    time_out_now = record_curTime.strftime("%Y-%m-%d %H:%M:%S")
                                    execute_query(connection, f'UPDATE StudentAttendance SET Time_Out = "{time_out_now}" WHERE Id = {student_id}')
                                    img_path = f"entry_picture/{student_id}_out_img.jpg"
                                    cv2.imwrite(img_path, image)
                                    if Time_In != None:
                                        execute_query(connection, f"""UPDATE Students
                                                                                                 SET Attendance_Count = {Attendance_Count + 1}
                                                                                                 WHERE Id = {student_id}""")
                                
                                else:
                                    intruder_attack = 0
                                    webcam[0:600, 600:424 + 600] = cutting_picture
                                    if curTime > result_time4:
                                        playAudio("sounds/Cutting.wav")
                                        now4 = datetime.now()
                                        current_time4 = now4.strftime("%H:%M:%S")
                                        result_time4 = datetime.combine(date.today(),time(int(current_time4[:2]),
                                                                    int(current_time4[3:5]),
                                                                    int(current_time4[6:8]))) + timedelta(seconds=5)
                                   
                            elif day_of_week == 4:
                                fri_query = f"SELECT * FROM Schedule WHERE ID = {student_id}"
                                my_result_date = str(read_query(connection, fri_query)[0][5])
                                my_result_date = time(hour=int(my_result_date[:2]), minute =int(my_result_date[3:5]), second =int(my_result_date[6:]))
                                time_now = now.time()
                                if time_now > my_result_date:
                                    playAudio("sounds/Face_Recognized_voice.wav")
                                    time_out_now = record_curTime.strftime("%Y-%m-%d %H:%M:%S")
                                    execute_query(connection, f'UPDATE StudentAttendance SET Time_Out = "{time_out_now}" WHERE Id = {student_id}')
                                    img_path = f"entry_picture/{student_id}_out_img.jpg"
                                    cv2.imwrite(img_path, image)
                                    if Time_In != None:
                                        execute_query(connection, f"""UPDATE Students
                                                                                                 SET Attendance_Count = {Attendance_Count + 1}
                                                                                                 WHERE Id = {student_id}""")
                                
                                else:
                                    intruder_attack = 0
                                    webcam[0:600, 600:424 + 600] = cutting_picture
                                    if curTime > result_time4:
                                        playAudio("sounds/Cutting.wav")
                                        now4 = datetime.now()
                                        current_time4 = now4.strftime("%H:%M:%S")
                                        result_time4 = datetime.combine(date.today(),time(int(current_time4[:2]),
                                                                    int(current_time4[3:5]),
                                                                    int(current_time4[6:8]))) + timedelta(seconds=5)
                            elif day_of_week == 5:
                                sat_query = f"SELECT * FROM Schedule WHERE ID = {student_id}"
                                my_result_date = str(read_query(connection, sat_query)[0][6])
                                my_result_date = time(hour=int(my_result_date[:2]), minute =int(my_result_date[3:5]), second =int(my_result_date[6:]))
                                time_now = now.time()
                                if time_now > my_result_date:
                                    playAudio("sounds/Face_Recognized_voice.wav")
                                    time_out_now = record_curTime.strftime("%Y-%m-%d %H:%M:%S")
                                    execute_query(connection, f'UPDATE StudentAttendance SET Time_Out = "{time_out_now}" WHERE Id = {student_id}')
                                    img_path = f"entry_picture/{student_id}_out_img.jpg"
                                    cv2.imwrite(img_path, image)
                                    if Time_In != None:
                                        execute_query(connection, f"""UPDATE Students
                                                                                                 SET Attendance_Count = {Attendance_Count + 1}
                                                                                                 WHERE Id = {student_id}""")
                                
                                else:
                                    intruder_attack = 0
                                    webcam[0:600, 600:424 + 600] = cutting_picture
                                    if curTime > result_time4:
                                        playAudio("sounds/Cutting.wav")
                                        now4 = datetime.now()
                                        current_time4 = now4.strftime("%H:%M:%S")
                                        result_time4 = datetime.combine(date.today(),time(int(current_time4[:2]),
                                                                    int(current_time4[3:5]),
                                                                    int(current_time4[6:8]))) + timedelta(seconds=5)
                                    
                    else:
                        webcam[0:600, 600:424 + 600]= guiInterFace[5]
                        student_img = images_text.index(student_id)
                        webcam[135:215 + 135, 688:245 + 688] = img_for_gui[student_img]
                        cv2.putText(webcam,f"{LastName}",(680, 412), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (255,255,255), 2)
                        cv2.putText(webcam,f"{Student_Id}",(680, 485), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6, (255,255,255), 2)
                        cv2.putText(webcam,f"{Program}",(680, 560), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.6, (255,255,255), 2)
                        intruder_attack = 0
                        
                except:
                    cvzone.cornerRect(webcam, bbox, rt=0, colorC=(255, 0, 0))
                    webcam[0:600, 601:423 + 601] = guiInterFace[3]
                    if curTime > result_time:
                        playAudio("sounds/Warning_voice.wav")
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        result_time = datetime.combine(date.today(),time(int(current_time[:2]),
                                                                int(current_time[3:5]),
                                                                int(current_time[6:8]))) + timedelta(seconds=5)
                        intruder_attack += 1
                        if intruder_attack == 5:
                            intrud_img = "intrud_img.jpg"
                            cv2.imwrite(intrud_img, image)
                            email_sender_security("clarencecarcuevas@gmail.com", "High Offiecer", "Security", intrud_img)
                            os.remove(intrud_img)
                            intruder_attack == 0
                else:
                    cvzone.cornerRect(webcam, bbox, rt=0)
                    cvzone.putTextRect(webcam,f"{FirstName}", (x2 -130, y2  + 70), thickness=3, scale = 2)    
               
    elif class_name == "Spoofing":
            webcam[0:600, 600: 424 + 600] = guiInterFace[1]
            if curTime > result_time2:
                now2 = datetime.now()
                current_time2 = now2.strftime("%H:%M:%S")
                result_time2 = datetime.combine(date.today(),time(int(current_time2[:2]),
                                            int(current_time2[3:5]),
                                            int(current_time2[6:8]))) + timedelta(seconds=5)
            
    else:
        webcam[0:600, 600: 424 + 600] = guiInterFace[0]      
    cv2.imshow("Face Recognition", webcam)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()

