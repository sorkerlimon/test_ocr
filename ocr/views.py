from multiprocessing import context
from django.shortcuts import render,redirect
from .models import Esr, Imageadd
from .ex import Detect_Text
from PIL import Image
import numpy as np
import cv2

import io
from django.core.files.storage import FileSystemStorage
from django.core.files.storage import Storage
from django.core.files.base import ContentFile
from google.cloud import vision




# Create your views here.
def test(request):
    return render(request,'test.html')



def index(request):


    if request.method == 'POST' and request.FILES['pic']:
        image = request.FILES['pic']
        

        de = Detect_Text(image)
        print(de)

        # # obj = Imageadd(image=image)
        # # obj.save()
        # return redirect('test')
    
    return render(request,'index.html')


    