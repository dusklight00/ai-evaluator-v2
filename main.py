from main.Evaluator import EVALUATOR

evaluator = EVALUATOR('C:/Program Files/Tesseract-OCR/tesseract.exe')

image_url = '<image-url-here>'
recognised_text = evaluator.recognize_word(image_url)
print(recognised_text)