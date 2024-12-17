import sys

import cv2
import time
from typing import List, Tuple
from src.utils import save_violation_report
from src.exception import CustomException
from src.components.face_recognitions import FaceRecognizer

from ultralytics import YOLO

import face_recognition


class MonitoringApp:
    def __init__(self, model_path, class_names: List[str], face_recognizer: FaceRecognizer):
        """
        Initialize the monitoring app.

        Args:
            model: Object detection model.
            class_names (List[str]): List of class names detected by the model.
            face_recognizer (FaceRecognizer): Instance of the face recognition system.
        """
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.face_recognizer = face_recognizer
        self.running = False
        self.detected_workers = []
        self.violations_list = []
        self.alerts_list = []


    def process_detections(self, results, frame, violations_list, alerts_list):
        """
        Process detections from the model and log any violations.

        Args:
            results: Model inference results.
            frame: Frame from the video stream.
            violations_list: List to store detected violations.
            alerts_list: List to store generated alerts.
        """
        for result in results:
            try:
                for box in result.boxes:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = self.class_names[class_id]

                    if "NO-" in class_name:
                        violation_text = f"Violation Detected: {class_name} at ({x_min}, {y_min})"
                        violations_list.append(violation_text)

                    # Draw bounding boxes
                    color = (0, 255, 0) if "NO-" not in class_name else (0, 0, 255)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Perform face recognition and annotate with "Person"
                recognized_faces = self.face_recognizer(frame)
                for face, _, (top, right, bottom, left) in recognized_faces:
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
                    cv2.putText(frame, face, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            except Exception as e:
                raise CustomException(e, sys)

    def log_violation(self, violation_text: str):
        """
        Log a violation.

        Args:
            violation_text (str): Text describing the violation.
        """
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logged_text = f"{current_time}: {violation_text}"
        self.violations_list.append(logged_text)

    def generate_report(self, filename: str = "violations_report.txt"):
        """
        Generate a violation report and save it to a file.

        Args:
            filename (str): Name of the report file.
        """
        try:
            save_violation_report(self.violations_list, filename)
            print(f"Report successfully saved to {filename}.")
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    class_names = [
        'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
        'NO-Safety Vest', 'Person', 'Safety Cone',
        'Safety Vest', 'machinery', 'vehicle'
    ]

    # Initialize FaceRecognizer with dummy data
    image = cv2.imread(r"C:\Users\ASUS\Desktop\Folders\Grad_project\known_faces\mohammad.jpg")
    location = face_recognition.face_locations(image)
    encoding = face_recognition.face_encodings(image, location)
    face_recognizer = FaceRecognizer([encoding], ["mohammad"])

    # Initialize MonitoringApp with YOLO model and class names
    app = MonitoringApp(model_path="best.pt", class_names=class_names, face_recognizer=face_recognizer)

    # Start video capture
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO model
            results = app.model(frame)

            # Process detections
            app.process_detections(results, frame, app.violations_list, app.alerts_list)

            # Display the frame
            cv2.imshow("Monitoring App", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Generate report
    app.generate_report()
