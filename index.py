import cv2
from preprocessor.detector import detect_words
from preprocessor.word_seg import find_optimal_central_line, find_word_line

image = cv2.imread("answer.png")
words = detect_words(image)
central_lines = find_optimal_central_line(words)
line = find_word_line(words[0], central_lines)

print(line)