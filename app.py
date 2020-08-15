#Importing necessary packages
import numpy as np
from flask import Flask, request
import pickle
from fastai.vision import *
import os
import json
import urllib.request
#Saving the working directory and model directory
cwd = os.getcwd()
path = cwd + '/model'

#Initializing the FLASK API
app = Flask(__name__)

#Loading the saved model using fastai's load_learner method
model = load_learner(path, 'export.pkl')

@app.route('/')
def handler():
    defaults.device = torch.device('cpu') # set device to cpu
    #image = request.files.get('image')

    img = open_image('./tmp/image.jpg')
    pred_class,pred_idx,outputs = model.predict(img)


    return(str(pred_class))
   

if __name__ == "__main__":
    app.run(debug=True)