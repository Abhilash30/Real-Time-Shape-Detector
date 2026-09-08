
import cv2



def to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def canny_edge_detection(frame, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(frame, low_threshold, high_threshold)
    return edges


def to_threshold(frame):
    blur = cv2.GaussianBlur(frame, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def dilate(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return dilated
def erode(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(binary, kernel, iterations=1)
    return eroded

def opening(binary):
    eroded = erode(binary)
    return dilate(eroded)

def closing(binary):
    dilated = dilate(binary)
    return erode(dilated)

def clean_binary(binary):
    opened = opening(binary)
    closed = closing(opened)
    return closed

def find_contours(binary):
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

