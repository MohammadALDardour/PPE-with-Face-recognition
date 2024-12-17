import customtkinter as ctk
from src.components.monitoring import MonitoringApp
from src.components.face_recognitions import FaceRecognizer
from src.utils import init_video_stream, get_frame, convert_image_to_tk
from ultralytics import YOLO
import face_recognition

# Initialize known faces
known_face_images = ["./known_faces/mohammad.jpg"]
known_face_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(img))[0] for img in known_face_images]
known_face_names = ["Mohammad"]

# Initialize global objects
face_recognizer = FaceRecognizer(known_face_encodings, known_face_names)
model = YOLO("./best.pt")
class_names = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
    'NO-Safety Vest', 'Person', 'Safety Cone',
    'Safety Vest', 'machinery', 'vehicle'
]

def main():
    app = ctk.CTk()
    app.title("Factory Security Dashboard")
    app.geometry("1920x1080")
    monitoring_app = MonitoringApp(model, class_names, face_recognizer)

    # GUI and button events
    def start_monitoring():
        monitoring_app.running = True
        update_video_feed()

    def stop_monitoring():
        monitoring_app.running = False

    def update_video_feed():
        if monitoring_app.running:
            frame = get_frame(video_stream)
            if frame is not None:
                img = convert_image_to_tk(frame)
                video_canvas.imgtk = img
                video_canvas.configure(image=img)
                results = model(frame, stream=True)
                monitoring_app.process_detections(results, frame, violations_list=[], alerts_list=[])
            app.after(10, update_video_feed)

    def capture_and_match():
        frame = get_frame(video_stream)
        recognized_faces = monitoring_app.recognize_faces(frame)
        for name, confidence, location in recognized_faces:
            monitoring_app.log_worker_recognition(name, confidence)

    # Configure GUI components
    video_frame = ctk.CTkFrame(app)
    video_frame.grid(row=0, column=0)
    video_canvas = ctk.CTkLabel(video_frame)
    video_canvas.pack()

    button_frame = ctk.CTkFrame(app)
    button_frame.grid(row=1, column=0)
    start_button = ctk.CTkButton(button_frame, text="Start Monitoring", command=start_monitoring)
    stop_button = ctk.CTkButton(button_frame, text="Stop Monitoring", command=stop_monitoring)
    capture_button = ctk.CTkButton(button_frame, text="Capture Face", command=capture_and_match)
    start_button.pack()
    stop_button.pack()
    capture_button.pack()

    video_stream = init_video_stream()
    app.mainloop()


if __name__ == "__main__":
    main()
