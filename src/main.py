import cv2
import os
from datetime import datetime
from dataset import save_sample
from dataset import create_dataset

from webcam import open_camera, get_frame
from preprocessing import to_grayscale
from preprocessing import to_threshold
from preprocessing import clean_binary
from preprocessing import find_contours
from preprocessing import canny_edge_detection
from features import contour_area
from features import contour_perimeter
from features import aspect_ratio
from features import circularity
from features import corner_count
from classifier import classify_shape
from features import bounding_box
import matplotlib.pyplot as plt
from dataframe import df

SAVE_DIR = "captures"
WINDOW_ORIGINAL = "Original (BGR)"
WINDOW_GRAY = "Grayscale"
WINDOW_BINARY = "Binary (Thresholded)"
WINDOW_CLOSING = "Closing (clean binary)"
WINDOW_CONTOURS = "Contours"
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    create_dataset()
    cap = open_camera()
    if cap is None:
        return

    print("Press 'q' to quit, 's' to save a frame.")
    frame_count = 0

    try:
        while True:
            ret, frame = get_frame(cap)
            if not ret:
                print("Error: failed to read frame from webcam.")
                break

            gray = to_grayscale(frame)
            #can = canny_edge_detection(gray)
            binary = to_threshold(gray)
            cleaned = clean_binary(binary)
            contours = find_contours(binary)
            display_frame = cleaned.copy()
            # cv2.imshow(WINDOW_ORIGINAL, frame)
            # cv2.imshow(WINDOW_GRAY, gray)
            # cv2.imshow(WINDOW_BINARY, binary)
            
            display_frame = frame.copy()

            min_area = 100
            max_area = 10000
            contours = [c for c in contours if min_area < contour_area(c) < max_area]

            for contour in contours:
                
                area = contour_area(contour)
                perimeter = contour_perimeter(contour)
                vertices = corner_count(contour)
                ratio = aspect_ratio(contour)
                circle = circularity(contour)

                shape = classify_shape(vertices, circle, ratio)
                print(f"Detected shape: {shape}, Vertices: {vertices}, Aspect Ratio: {ratio:.2f}, Circularity: {circle:.2f}, Area: {area:.2f}, Perimeter: {perimeter:.2f}")
                x, y, w, h = bounding_box(contour)
                feature_vector = [area, perimeter, vertices, ratio, circle]

                save_sample(feature_vector, shape)
                # Draw contour
                cv2.drawContours(
                    display_frame,
                    [contour],
                    -1,
                    (0, 255, 255),
                    2
                )

                # bounding box
                cv2.rectangle(
                    display_frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # shape name
                cv2.putText(
                    display_frame,
                    shape,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            
            cv2.imshow(WINDOW_CLOSING, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting.")
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(SAVE_DIR, f"frame_{timestamp}.png")
                cv2.imwrite(filename, display_frame)
                frame_count += 1
                print(f"Saved: {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Done. Saved {frame_count} frame(s) to '{SAVE_DIR}/'.")


if __name__ == "__main__":
    main()
