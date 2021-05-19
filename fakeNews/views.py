from django.shortcuts import render
from . import ml

# Create your views here.
def index(request):    
    if request.method == "POST":
        new_text = ml.classifier(title=request.POST.get("text1", None), text=request.POST.get("text2", None))
    else:
        return render(request, 'fakeNews/index.html', {'result1': '', 'result2': '', 'accuracy': ml.get_accuracy()})
    return render(request, 'fakeNews/index.html', {'result1': new_text[0], 'result2': new_text[1], 'accuracy': ml.get_accuracy()})
    