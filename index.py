import os
import cv2
from core.detector import detect_words, extract_word
from core.word_recognition import recognize_word_api
from utils import cv2_image_to_base64
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

TR_OCR_API = os.getenv("TR_OCR_API")

image = cv2.imread("answer.png")
words = detect_words(image)

for index, word in tqdm(enumerate(words)):
    extracted_word = extract_word(image, word)
    extracted_word_base64 = cv2_image_to_base64(extracted_word)
    word = recognize_word_api(extracted_word_base64, TR_OCR_API)
    words[index]["word"] = word