from flask import Flask, render_template, request, jsonify
from werkzeug import secure_filename
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2 as cv
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image as img



age_dict={
    0:'below_20',
    1:'20-30',
    2:'30-40',
    3:'40-50',
    4:'above_50'
}

gender_dict = {
    0:'M',
    1:'F'
}
ethnicity_dict={
    0:'arab',
    1:'asian',
    2:'black',
    3:'hispanic',
    4:'indian',
    5:'white'
}
emotion_dict={
    0:'angry',
    1:'happy',
    2:'neutral',
    3:'sad'
}



app = Flask(__name__)

global name
@app.route('/')
def hello():
    return render_template('index.html')



@app.route('/uploader', methods = ['GET', 'POST'])
def status():
    if request.method == 'POST':
        f = request.files['file']
        f.filename = 'image.jpg'
        f.save(secure_filename(f.filename))
        return render_template('status.html')



@app.route('/predict')
def predict():
    image = img.load_img('image.jpg',target_size=(70,70))
    image = img.img_to_array(image)
    image = image/255
    image = image.reshape(1,70,70,3)
    #Predicting
    g_pred = gender_pred(image)
    e_pred = ethnicity_pred(image)
    a_pred = age_pred(image)
    emo_pred = emotions_pred(image)
    pred_dict={
    "Gender" : g_pred,
    "Age" : a_pred,
    "Ethnicity" : e_pred,
    "Emotions": emo_pred
    }

    return render_template('predictions.html',posts = pred_dict)


def gender_pred(image):
    with graph.as_default():
        gen = gender_model.predict(image).argmax()
        return gender_dict[gen]


def emotions_pred(image):
    with graph.as_default():
        emotion = emotions_model.predict(image).argmax()
        return emotion_dict[emotion]


def ethnicity_pred(image):
    with graph.as_default():
        eth = ethnicity_model.predict(image).argmax()
        return ethnicity_dict[eth]

def age_pred(image):
    with graph.as_default():
        age = age_model.predict(image).argmax()
        return age_dict[age]

if __name__ == '__main__':
    global model, graph
    ethnicity_model = load_model('ethnicity_model.h5')
    gender_model = load_model('gender_model.h5')
    age_model = load_model('age_model.h5')
    emotions_model = load_model('emotions_model.h5')
    graph = tf.get_default_graph()
    app.run(host = '0.0.0.0',port=5005)