import base64
import cv2

def crop_image(image, x, y, width, height):
    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)
    return image[y:y+height, x:x+width]

def cv2_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')