from django.shortcuts import render
from main.dummy_evaluator import dummy_evalute

# Create your views here.
def homepage(request):
    image_url = request.POST.get('image_url', '')
    if image_url:
        try:
            result = dummy_evalute(image_url)
            return render(request, 'main_app/index.html', {'result': result})
        except:
            return render(request, 'main_app/index.html')
        
    return render(request, 'main_app/index.html')