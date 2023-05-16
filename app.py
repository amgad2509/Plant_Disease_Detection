#from gettext import install

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.backend import argmax
import efficientnet.tfkeras as efn
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)


model = tf.keras.models.load_model('best_model.h5',compile=False)


def get_className(classNo):
  if classNo==0:
      return "Apple___Apple_scab"
  elif classNo==1:
      return "Apple___Black_rot"
  elif classNo==2: 
      return "Apple___Cedar_apple_rust"
  elif classNo==3:
      return "Apple___healthy"
  elif classNo==4:
      return "Blueberry___healthy"  
  elif classNo==5:
      return "Cherry_(including_sour)___Powdery_mildew"
  elif classNo==6:
      return "Cherry_(including_sour)___healthy"
  elif classNo==7:
      return "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
  elif classNo==8:
      return "Corn_(maize)___Common_rust_"
  elif classNo==9:
      return "Corn_(maize)___Northern_Leaf_Blight"
  elif classNo==10:
      return "Corn_(maize)___healthy"
  elif classNo==11:
      return "Grape___Black_rot"
  elif classNo==12:
      return "Grape___Esca_(Black_Measles)"
  elif classNo==13:
      return "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"
  elif classNo==14:
      return "Grape___healthy"
  elif classNo==15:
      return "Orange___Haunglongbing_(Citrus_greening)"
  elif classNo==16:
      return "Peach___Bacterial_spot"
  elif classNo==17:
      return "Peach___healthy"
  elif classNo==18:
      return "Pepper,_bell___Bacterial_spot"
  elif classNo==19:
      return "Pepper,_bell___healthy"
  elif classNo==20:
      return "Potato___Early_blight"
  elif classNo==21:
      return "Potato___Late_blight"
  elif classNo==22:
      return "Potato___healthy"
  elif classNo==23:
      return "Raspberry___healthy"
  elif classNo==24:
      return "Soybean___healthy"
  elif classNo==25:
      return "Squash___Powdery_mildew"
  elif classNo==26:
      return "Strawberry___Leaf_scorch"
  elif classNo==27:
      return "Strawberry___healthy"
  elif classNo==28:
      return "Tomato___Bacterial_spot"
  elif classNo==29:
      return "Tomato___Early_blight"
  elif classNo==30:
      return "Tomato___Late_blight"
  elif classNo==31:
      return "Tomato___Leaf_Mold"
  elif classNo==32:
      return "Tomato___Septoria_leaf_spot"
  elif classNo==33:
      return "Tomato___Spider_mites_Two-spotted_spider_mite"
  elif classNo==34:
      return "Tomato___Target_Spot"
  elif classNo==35:
      return "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
  elif classNo==36:
      return "Tomato___Tomato_mosaic_virus"
  elif classNo==37:
      return "Tomato___healthy"

def getResult(img):
    imge = cv2.imread(img)
    img = Image.fromarray(imge)
    img = img.resize((224,224))
    img = np.array(img)
    #print(img)

    
    input_img = np.expand_dims(img,axis=0)
    predictions = model.predict(input_img)
    classes_x=argmax(predictions,axis=1)
    #print(classes_x)
    return classes_x


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('main_page.html')

@app.route('/admin_page', methods=['GET'])
def admin_page():
    return render_template('admin_page.html')

@app.route('/take_photo', methods=['GET'])
def take_photo():
    return render_template('take_photo.html')


@app.route('/predict', methods = ['POST'])
def predict():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    value=getResult(file_path)
    result=get_className(value) 
    return render_template('take_photo.html', prediction_text= result)


if __name__ == '__main__':
    app.run(debug=True)