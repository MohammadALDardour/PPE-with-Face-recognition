import customtkinter as ctk
import tkinter.messagebox as tkmb
import cv2
from PIL import Image, ImageTk
import csv
import time
from ultralytics import YOLO
import math
import cvzone
import tkinter as tk
from tkinter import messagebox
from tkinter import Label
import os

from deepface import DeepFace  # Import DeepFace for face matching
# Globals
video_stream = None
running = False
detected_workers = []
model = YOLO("./best.pt")
class_names = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
    'NO-Safety Vest', 'Person', 'Safety Cone',
    'Safety Vest', 'machinery', 'vehicle'
]
known_faces = [
    {
        "name": "mohammad",
        "role": "Engineer",
        "blood_type": "O+",
        "age":30,
        "sector": "Manufacturing",
        "image_path": "./known_faces/mohammad.JPG"  # Path to the known face image
    },
    {
        "name": "Jane Smith",
        "role": "Supervisor",
        "blood_type": "A-",
        "age":30,
        "sector": "Quality Control",
        "image_path": "./known_faces/hareth.jpg"
    }
]

# Initialize GUI
app = ctk.CTk()
app.title("Factory Security Dashboard")
app.geometry("1920x1080")
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Frames and Widgets
video_frame = ctk.CTkFrame(app, corner_radius=15)
video_frame.grid(row=0, column=0, rowspan=3, padx=20, pady=20, sticky="nsew")
video_label = ctk.CTkLabel(video_frame, text="Real-Time Feed", font=("Helvetica", 16))
video_label.pack(pady=10)
video_canvas = ctk.CTkLabel(video_frame, text="")
video_canvas.pack(padx=10, pady=10, expand=True, fill="both")


details_frame = ctk.CTkFrame(app, corner_radius=15)
details_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew" )
details_label = ctk.CTkLabel(details_frame, text="Person Details", font=("Helvetica", 16))
details_label.pack(pady=10)
person_image = ctk.CTkLabel(details_frame, text="No Image")
person_image.pack(pady=10)
person_info = ctk.CTkLabel(details_frame, text="No Person Detected", font=("Helvetica", 14),height=200)
person_info.pack(pady=10)



violations_frame = ctk.CTkFrame(app, corner_radius=15)
violations_frame.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")
violations_label = ctk.CTkLabel(violations_frame, text="Violations", font=("Helvetica", 16))
violations_label.pack(pady=10)
violations_list = ctk.CTkScrollableFrame(violations_frame, height=200)
violations_list.pack(padx=10, pady=10, expand=True, fill="both")

alerts_frame = ctk.CTkFrame(app, corner_radius=15)
alerts_frame.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
alerts_label = ctk.CTkLabel(alerts_frame, text="Alerts", font=("Helvetica", 16))
alerts_label.pack(pady=10)
alerts_list = ctk.CTkScrollableFrame(alerts_frame, height=200)
alerts_list.pack(padx=10, pady=10, expand=True, fill="both")

button_frame = ctk.CTkFrame(app, corner_radius=15)
button_frame.grid(row=3, column=0, columnspan=2, pady=20, sticky="ew")
start_button = ctk.CTkButton(button_frame, text="Start Monitoring", command=lambda: start_monitoring())
start_button.pack(side="left", padx=10, pady=10)
stop_button = ctk.CTkButton(button_frame, text="Stop Monitoring", command=lambda: stop_monitoring())
stop_button.pack(side="left", padx=10, pady=10)

capture_button = ctk.CTkButton(button_frame, text="capture", command=lambda: capture_and_match_face())
capture_button.pack(side="left", padx=10, pady=10)

report_button = ctk.CTkButton(button_frame, text="Generate Report", command=lambda: generate_report())
report_button.pack(side="right", padx=10, pady=10)
close_button = ctk.CTkButton(button_frame, text="Exit", command=lambda: on_close())
close_button.pack(side="right", padx=10, pady=10)

app.grid_rowconfigure(0, weight=1)
app.grid_rowconfigure(1, weight=1)
app.grid_rowconfigure(2, weight=1)
app.grid_rowconfigure(3, weight=0)
app.grid_columnconfigure(0, weight=3)
app.grid_columnconfigure(1, weight=1)


def init_video_stream():
    global video_stream
    video_stream = cv2.VideoCapture(0)
    video_stream.set(3, 1280)
    video_stream.set(4, 720)


def start_monitoring():
    global running
    running = True
    update_video_feed()


def stop_monitoring():
    global running
    running = False
    video_canvas.configure(image="", text="Monitoring Stopped")
    clear_text_sections()


def update_video_feed():
    global running
    if running and video_stream:
        ret, frame = video_stream.read()
        if ret:
            results = model(frame, stream=True)
            process_detections(results, frame)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            video_canvas.imgtk = imgtk
            video_canvas.configure(image=imgtk,text="")
        app.after(10, update_video_feed)


def process_detections(results, frame):
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = box.conf[0]
            label = class_names[cls]
            color = (0, 255, 0) if "NO-" not in label else (0, 0, 255)
            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cvzone.putTextRect(frame, f"{label} {conf:.2f}", (x1, y1 - 10), scale=1, colorR=color)
                if label == "Person":
                    # update_person_details(label, Image.fromarray(frame[y1:y2, x1:x2]))
                    pass
                elif "NO-" in label:
                    log_violation("Unknown", label)
                    add_alert(f"Alert: {label} detected!")

def update_person_details(worker):
    person_info.configure(text=f"Name: {worker['name']}\n Blood_type: {worker['blood_type']}\n Sector:{worker['sector']}")
    # Load and display the worker's image
    image = Image.open(worker["image_path"])
    image = image.resize((100, 100))
    worker_img = ImageTk.PhotoImage(image)
    person_image.configure(image=worker_img,text="")
    person_image.image = worker_img


def log_violation(worker, violation):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    new_violation = ctk.CTkLabel(violations_list, text=f"{current_time}: {worker} - {violation}")
    new_violation.pack(anchor="w", padx=10, pady=5)
    detected_workers.append({"Time": current_time, "Name": worker, "Violation": violation})


def add_alert(message):
    new_alert = ctk.CTkLabel(alerts_list, text=message)
    new_alert.pack(anchor="w", padx=10, pady=5)


def clear_text_sections():
    person_image.configure(image="", text="No Image")
    person_info.configure(text="No Person Detected")
    for frame in (violations_list, alerts_list):
        for widget in frame.winfo_children():
            widget.destroy()


def generate_report():
    if not detected_workers:
        tkmb.showinfo("No Data", "No violations detected yet.")
        return
    with open("violations_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Time", "Name", "Violation"])
        writer.writeheader()
        writer.writerows(detected_workers)
    tkmb.showinfo("Report Generated", "Violations report saved as 'violations_report.csv'.")




def get_frame():
        if video_stream.isOpened():
            ret, frame =video_stream.read()
            if ret:
                return frame
        return None

def capture_and_match_face():
    frame = get_frame()
    if frame is not None:
        # Save the captured frame temporarily
        temp_captured_path = "./captured_frame.jpg"
        cv2.imwrite(temp_captured_path, frame)

        # Iterate over known faces and match
        for worker in known_faces:
            try:
                result = DeepFace.verify(
                    img1_path=temp_captured_path,  # Captured frame
                    img2_path=worker["image_path"],  # Known face path
                    model_name='VGG-Face',
                    enforce_detection=False
                )
                if result["verified"]:
                    update_person_details(worker)
                    os.remove(temp_captured_path)
                    return
            except Exception as e:
                print(f"Error during face verification: {e}")
                tkmb.showerror("Error", "Error during face verification.")

    os.remove(temp_captured_path)
    tkmb.showinfo("No Match", "No matching worker found.")
    

def on_close():
    if video_stream:
        video_stream.release()
    app.destroy()

init_video_stream()
app.mainloop()

