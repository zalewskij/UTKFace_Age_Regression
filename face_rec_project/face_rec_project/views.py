from django.shortcuts import render


# renders the initial menu
def index(request):    
    return render(request, 'index.html')
