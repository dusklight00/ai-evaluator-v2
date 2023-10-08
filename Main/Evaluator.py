import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from word_detector import prepare_img, detect, sort_line
import pytesseract
import base64
import cv2
import requests
import json

class EVALUATOR:
    
    def __init__(self,pytesseract_tesseract_cmd_path):
        self.pytesseract_tesseract_cmd_path = pytesseract_tesseract_cmd_path
        pass
    
    # UTILS
    def crop_image(self, image, x, y, width, height):
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        return image[y:y+height, x:x+width]
    
    def cv2_image_to_base64(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def base64_to_cv2_image(self, image_base64):
        image_bytes = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)
        return image
    
    #  DONE 1: Implement find_optimal_k AND find_optimal_cluster
    def find_optimal_k(self, data, max_k):
        best_k = None
        best_combined_metric = -np.inf
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(data.reshape(-1, 1))
            
            silhouette = silhouette_score(data.reshape(-1, 1), kmeans.labels_)
            davies_bouldin = davies_bouldin_score(data.reshape(-1, 1), kmeans.labels_)
            inertia = kmeans.inertia_
            
            epsilon = 1e-10
            combined_metric = silhouette - davies_bouldin + (1 / (inertia + epsilon))
            
            if combined_metric > best_combined_metric:
                best_combined_metric = combined_metric
                best_k = k
                
        return best_k
    
    def find_optimal_cluster(self, data):
        MAX_CLUSTER = 10
        data_np = np.array(data)
        best_k = self.find_optimal_k(data_np, MAX_CLUSTER)
        return best_k
    
    # DONE 2: Implement find_optimal_central_line AND find_word_line 
    def find_num_lines(self, words):
        y_coords = [ word["y"] for word in words ]
        clusters = self.find_optimal_cluster(y_coords)
        return clusters
    
    def find_optimal_central_line(self, words):
        y_coords = np.array([ word["y"] for word in words ])
        num_lines = self.find_num_lines(words) 

        kmeans = KMeans(n_clusters=num_lines)
        kmeans.fit(y_coords.reshape(-1, 1))  

        cluster_centers = kmeans.cluster_centers_
        cluster_centers = np.sort(cluster_centers, axis=0)

        return list(cluster_centers.reshape(-1))
    
    def find_word_line(self, word, central_lines):
        y_coord = word["y"]
        distances = [ abs(y_coord - line) for line in central_lines ]
        return np.argmin(distances)
    
    def detect_words(self, image):
        prepared_image = prepare_img(image, 200)
        detections = detect(prepared_image, kernel_size=25, sigma=11, theta=7, min_area=50)
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
    
    def extract_word(self, image, word):
        x = word["x"]
        y = word["y"]
        width = word["width"]
        height = word["height"]
        return self.crop_image(image, x, y, width, height)
    
    
    def recognize_word(self, image_base64):
        pytesseract.pytesseract.tesseract_cmd = self.pytesseract_tesseract_cmd_path
        image = self.base64_to_cv2_image(image_base64)
        word = pytesseract.image_to_string(image)
        return word
    
    def trOCR_recognize_word_api(self, image_base64,base_url):
        url = base_url + "/recognize_word"
        data = { "image_base64": image_base64 }
        headers = {'ngrok-skip-browser-warning': 'true'}
        response = requests.post(url, data=json.dumps(data), headers=headers)
        response = response.json()
        return response["word"]
    
    # DONE 