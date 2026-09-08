def classify_shape(vertices, circularity, aspect_ratio):
    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        if 0.80 <= aspect_ratio <= 1.5:
            return "Square"
        else:
            return "Rectangle"
    elif vertices > 6:
        if circularity > 0.8:
            return "Circle"
        