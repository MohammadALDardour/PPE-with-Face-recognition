import cv2
import cvzone

import time
from typing import List, Tuple

from src.utils import save_violation_report

from src.exception import CustomException

from ultralytics import YOLO

import face_recognition
from face_recognitions import FaceRecognizer


class DetectEPP:
    def __init__(self, model_path, classNames, face_recognizer):
        self.model = YOLO(model_path)
        self.classNames = classNames
        self.face_recognizer = face_recognizer
        self.violations_list = []
    

    def process_detections(self):
        cap = cv2.VideoCapture(r"C:\Users\ASUS\Desktop\Folders\Grad_project\src\components\videos\videotest2.webm")
        cap.set(cv2.CAP_PROP_FPS, 20)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            _, frame = cap.read()
            faces_info = self.face_recognizer.recognize_faces(frame)

            for name, _, (top, right, bottom, left) in faces_info:
                if name == "Unknown":
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)
                cv2.putText(frame, name, (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  
                cv2.rectangle(frame, (left, top), (right, bottom), color)

                self.epp_process(frame)

                cv2.imshow("cap", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()        


    def epp_process(self, frame):
        epp = self.model(frame)
        
        for result in epp:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = self.classNames[class_id]

                if "NO-" in class_name:
                    violation_text = f"Violation Detected: {class_name} at ({x_min}, {y_min})"
                    self.violations_list.append(violation_text)

                # Draw bounding boxes
                color = (0, 255, 0) if "NO-" not in class_name else (0, 0, 255)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


if __name__ == "__main__":
    class_names = [
        'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
        'NO-Safety Vest', 'Safety Cone',
        'Safety Vest', 'machinery', 'vehicle'
    ]

    image = cv2.imread(r"C:\Users\ASUS\Desktop\Folders\Grad_project\known_faces\Ali.png")
    location = face_recognition.face_locations(image)
    encoding = face_recognition.face_encodings(image, location)
    
    recognizer = FaceRecognizer(encoding, "Ali")
    obj = DetectEPP("best.pt", class_names, recognizer)

    obj.process_detections()