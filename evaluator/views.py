from django.shortcuts import render
from main.Evaluator import EVALUATOR

tesseract_path = '<path-to-tesseract-exe>'
# Create your views here.
def index(request):
    image_url = request.POST.get('image_url', '')
    evaluator = EVALUATOR(pytesseract_tesseract_cmd_path=tesseract_path)
    if image_url != '':
        try:
            recognised_text = evaluator.recognize_word(image_url)
            return render(request, 'evaluator/index.html', {'recognised_text': recognised_text})
        except Exception as e:
            return render(request, 'evaluator/index.html', {'error': e})
        
    return render(request, 'evaluatorindex.html')