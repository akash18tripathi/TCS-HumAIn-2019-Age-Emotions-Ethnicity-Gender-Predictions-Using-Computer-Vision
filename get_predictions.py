from flask import Flask, render_template, request, jsonify
from werkzeug import secure_filename
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2 as cv
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image as img
import cv2
import matplotlib.pyplot as plt
import shutil
from gevent.pywsgi import WSGIServer



age_dict={
    0:'below_20',
    1:'20-30',
    2:'30-40',
    3:'40-50',
    4:'above_50'
}

gender_dict = {
    0:'Male',
    1:'Female'
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
app.debug=True
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
    
    image = img.load_img('image.jpg')
    image = np.array(image,dtype='uint8')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.5,5)
    for (x,y,w,h) in faces:
        roi_color = image[y:y+h,x:x+w]
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imwrite('image.jpg',image)
        cv2.imwrite('face.jpg',roi_color)
    #image = img.img_to_array(image)
    image = img.load_img('face.jpg',target_size=(25,25))
    plt.imshow(image)
    image = img.img_to_array(image)
    image = image/255
    image = image.reshape(1,25,25,3)
    #Predicting
    g_pred = gender_pred(image)
    e_pred = ethnicity_pred(image)
    a_pred = age_pred(image)
    emo_pred = emotions_pred(image)
    image = img.load_img('image.jpg')
    image = np.array(image,dtype='uint8')
    #cv2.putText(image,"Age:"+str(a_pred)+", Gender:  "+str(g_pred)+",Emotion: "+str(emo_pred)+",Ethnicity: "+str(e_pred),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),1,cv2.LINE_AA)
    cv2.imwrite('image.jpg',image)
    shutil.copy('image.jpg','static/')
    pred_dict={
    "Gender" : g_pred,
    "Age" : a_pred,
    "Ethnicity" : e_pred,
    "Emotions": emo_pred,
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
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    graph = tf.get_default_graph()
    http_server = WSGIServer(('', 5005), app)
    http_server.serve_forever()
    
