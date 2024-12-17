import cv2
from PIL import Image, ImageTk


def save_violation_report(violations, output_file):
    import csv
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Time", "Name", "Violation"])
        writer.writeheader()
        writer.writerows(violations)

        
def init_video_stream():
    """Initialize the video stream."""
    video_stream = cv2.VideoCapture(0)
    video_stream.set(3, 1280)
    video_stream.set(4, 720)
    return video_stream


def get_frame(video_stream):
    """Capture a single frame from the video stream."""
    if video_stream.isOpened():
        ret, frame = video_stream.read()
        if ret:
            return frame
    return None


def convert_cv_to_image(frame):
    """Convert a CV frame to a PIL Image."""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def convert_image_to_tk(image):
    """Convert a PIL image to a Tkinter-compatible image."""
    return ImageTk.PhotoImage(image=image)




if __name__ == "__main__":

    cap = cv2.VideoCapture(r"C:\Users\ASUS\Desktop\Folders\Grad_project\src\components\videos\videotest.webm")

    while True:
        _, frame = cap.read()

        cv2.imshow("cap", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break
    cap.release()

    cv2.destroyWindow()    
