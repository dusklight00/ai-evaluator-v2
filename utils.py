import base64
import cv2
import numpy as np

def crop_image(image, x, y, width, height):
    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)
    return image[y:y+height, x:x+width]

def cv2_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_cv2_image(image_base64):
    image_bytes = base64.b64decode(image_base64)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)
    return image