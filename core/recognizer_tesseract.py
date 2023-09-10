import pytesseract
from utils import base64_to_cv2_image

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def recognize_word(image_base64):
    image = base64_to_cv2_image(image_base64)
    word = pytesseract.image_to_string(image)
    return word
