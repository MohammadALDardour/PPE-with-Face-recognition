import face_recognition

import numpy as np

from typing import List, Tuple

from src.exception import CustomException

import sys
import cv2


class IFaceRecognizer:
    """
    Interface for face recognition implementations.
    """
    def recognize_faces(self, frame: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Recognize faces in the given frame.

        Args:
            frame (np.ndarray): The frame containing faces.

        Returns:
            List[Tuple[str, float, Tuple[int, int, int, int]]]: List of tuples containing name, confidence, and location.
        """
        raise NotImplementedError


class FaceRecognizer(IFaceRecognizer):
    """
    Concrete implementation of IFaceRecognizer using face_recognition library.
    """

    def __init__(self, known_face_encodings: List[np.ndarray], known_face_names: List[str]):
        """
        Initialize the FaceRecognizer.

        Args:
            known_face_encodings (List[np.ndarray]): List of known face encodings.
            known_face_names (List[str]): List of known face names.
        """
        self.__known_face_encodings = known_face_encodings
        self.__known_face_names = known_face_names


    def recognize_faces(self, frame: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Recognize faces in the given frame.

        Args:
            frame (np.ndarray): The frame containing faces.

        Returns:
            List[Tuple[str, float, Tuple[int, int, int, int]]]: List of tuples containing name, confidence, and location.
        """
        # Detect faces in the frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        recognized_faces = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            try:
                matches = face_recognition.compare_faces(self.__known_face_encodings, face_encoding)
                name = "Unknown"
                confidence = 0.0

                face_distances = face_recognition.face_distance(self.__known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.__known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]

                recognized_faces.append((name, confidence, (top, right, bottom, left)))
            except Exception as e:
                raise CustomException(e, sys)
            
        return recognized_faces
    
    

if __name__ == "__main__":

    image = cv2.imread(r"C:\Users\ASUS\Desktop\Folders\Grad_project\known_faces\mohammad.jpg")
    location = face_recognition.face_locations(image)

    encoding = face_recognition.face_encodings(image, location)

    obj = FaceRecognizer(encoding, "mohammad")
    cap = cv2.VideoCapture(0)

    while True:
        susses, frame = cap.read()
        if not susses:
            print("not susses")
            break
        faces = obj.recognize_faces(frame)

        for name, _, (top, right, bottom, left) in faces:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0))
            cv2.putText(frame, name, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.imshow("cap", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()