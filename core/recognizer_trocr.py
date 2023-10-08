import requests
import json

def recognize_word_api(image_base64, base_url):
    url = base_url + "/recognize_word"
    data = { "image_base64": image_base64 }
    headers = {'ngrok-skip-browser-warning': 'true'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    response = response.json()
    return response["word"]