from word_detector import prepare_img, detect, sort_line

def detect_words(image):
    prepared_image = prepare_img(image, 200)

    detections = detect(
        prepared_image,
        kernel_size=25,
        sigma=11,
        theta=7,
        min_area=100
    )

    line = sort_line(detections)[0]

    rectangles = []
    for word in line:
        PADDING = 10
        x_cord = word.bbox.x - (PADDING / 2)
        y_cord = word.bbox.y - (PADDING / 2)
        width = word.bbox.w + (PADDING / 2)
        height = word.bbox.h + (PADDING / 2)
        rect = {
            "x": x_cord,
            "y": y_cord,
            "width": width,
            "height": height
        }
        rectangles.append(rect)
    
    return rectangles