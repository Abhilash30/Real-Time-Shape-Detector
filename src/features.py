import cv2
import numpy as np


def contour_area(contour):
    return cv2.contourArea(contour)


def contour_perimeter(contour):
    return cv2.arcLength(contour, True)


def bounding_box(contour):
    return cv2.boundingRect(contour)


def aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)

    if h == 0:
        return 0

    return w / h


def circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return 0

    return (4 * np.pi * area) / (perimeter ** 2)


def corner_count(contour):
    perimeter = cv2.arcLength(contour, True)

    epsilon = 0.05 * perimeter

    approximation = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )

    return len(approximation)