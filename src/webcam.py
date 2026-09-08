import cv2

CAMERA_INDEX = 0


def open_camera(index=CAMERA_INDEX):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: could not open webcam at index {index}")
        return None
    return cap


def get_frame(cap):
    return cap.read()
