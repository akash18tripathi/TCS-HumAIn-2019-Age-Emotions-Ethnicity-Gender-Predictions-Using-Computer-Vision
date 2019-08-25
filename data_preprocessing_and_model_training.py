#IMPORTING MODULES
import os
import numpy as np
import json
from PIL import Image
import requests
import cv2
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.utils import shuffle
from keras.preprocessing import image as img
from keras.models import load_model
import shutil
from keras.preprocessing.image import ImageDataGenerator


#=====================================================LOADING JSON FILE=============================
with open('Face_Recognition.json', 'r') as infile:
    # Variable for building our JSON block
    json_block = []
    for line in infile:
        # Add the line to our JSON block
        json_block.append(line)
        # Check whether we closed our JSON block
        if line.startswith('}'):
            # Do something with the JSON dictionary
            json_dict = json.loads(''.join(json_block))
            print(json_dict)
            # Start a new block
            json_block = []
       
  

      
#======================Extracting the facial area using the given co-ordinates and saving(writing) them.=============
            
#INTERNET CONNECTION REQUIRED.
for i in tqdm(range(121)):
    try:
        js = json.loads(json_block[i].replace("'", "\""))
        image = Image.open(requests.get(js['content'], stream=True).raw)
        image = np.asarray(image)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        for d in js['annotation']:
            width = d['imageWidth']
            height = d['imageHeight']
            x1 = int(width*d['points'][0]['x'])
            y1 = int(height*d['points'][0]['y'])
            x2 = int(width*d['points'][1]['x'])
            y2 = int(height*d['points'][1]['y'])
            roi = gray[y1:y2,x1:x2]
            s = d['label'][0]
            for k in range(1,len(d['label'])):
                s = s+" "+ d['label'][k]
            cv2.imwrite(s+'.jpg',roi)
    except:
        continue
  
#=================================MAKING EMPTY DIRECTORIES AS SPECIFIED BELOW:======================
        
#Before Proceeding, make a directory as follows:
        # ->dataset
        #       ->age
        #           ->below_20
        #           ->20-30
        #           ->30-40
        #           ->40-50
        #           ->above_50
        #
        #       ->emotions
        #           ->angry
        #           ->happy
        #           ->neutral
        #           ->sad
        #       
        #       ->ethnicity
        #           ->arab
        #           ->asian
        #           ->black
        #           ->hispanic
        #           ->indian
        #           ->white
        #
        #       ->gender
        #           ->M
        #           ->F
    
    
    
#===================================DISTRIBUTE THE IMAGES ACCORDING TO THEIR RESPECTIVE CATEGORIES, BY LABELS.=====================
        
for filename in os.listdir('./'):
    if filename.endswith('.jpg'):
        if 'below20' in filename:
            shutil.copy(filename,'./dataset/age/below_20/')
        if '20_30' in filename:
            shutil.copy(filename,'./dataset/age/20-30/')
        if '30_40' in filename:
            shutil.copy(filename,'./dataset/age/30-40/')
        if '40_50' in filename:
            shutil.copy(filename,'./dataset/age/40-50/')
        if 'above_50' in filename:
            shutil.copy(filename,'./dataset/age/above_50/')
        if 'Angry' in filename:
            shutil.copy(filename,'./dataset/emotions/angry/')
        if 'Happy' in filename:
            shutil.copy(filename,'./dataset/emotions/happy/')
        if 'Neutral' in filename:
            shutil.copy(filename,'./dataset/emotions/neutral/')
        if 'Sad' in filename:
            shutil.copy(filename,'./dataset/emotions/sad/')
        if 'Arab' in filename:
            shutil.copy(filename,'./dataset/ethnicity/arab/')
        if 'Asian' in filename:
            shutil.copy(filename,'./dataset/ethnicity/asian/')
        if 'Black' in filename:
            shutil.copy(filename,'./dataset/ethnicity/black/')
        if 'Hispanic' in filename:
            shutil.copy(filename,'./dataset/ethnicity/hispanic/')
        if 'Indian' in filename:
            shutil.copy(filename,'./dataset/ethnicity/indian/')
        if 'White' in filename:
            shutil.copy(filename,'./dataset/ethnicity/white/')
        if 'Female' in filename:
            shutil.copy(filename,'./dataset/gender/F/')
        if 'Male' in filename:
            shutil.copy(filename,'./dataset/gender/M/')

#===============================================AGE TRAINING MODEL DATA PREPROCESSING=============================================================================
            

#AGE TRAINING MODEL DATA PREPROCESSING          
age_train_x = []
age_train_y = []
age_dict={
    'below_20':0,
     '20-30':1,
    '30-40':2,
    '40-50':3,
    'above_50':4
}
for item in os.listdir('./dataset/age'):
    for p in os.listdir('./dataset/age/'+str(item)):
        try:
            image = keras.preprocessing.image.load_img('./dataset/age/'+str(item)+"/"+str(p),target_size=(70,70))
            image = keras.preprocessing.image.img_to_array(image)
            image = image/255
            age_train_x.append(image)
            age_train_y.append(age_dict[item])
        except:
            continue
#Converting into array
age_train_x = np.asarray(age_train_x)
age_train_y = np.asarray(age_train_y)
#Categorical Splitting
age_train_y = keras.utils.to_categorical(age_train_y)
#Shuffling the training data
age_train_x, age_train_y = shuffle(age_train_x,age_train_y)



#===============================================GENDER TRAINING MODEL DATA PREPROCESSING=================================================================================


gender_train_x = []
gender_train_y=[]

gender_dict = {
    'M':0,
    'F':1
}

for item in os.listdir('./dataset/gender'):
    for p in os.listdir('./dataset/gender/'+str(item)):
        try:
            image = keras.preprocessing.image.load_img('./dataset/gender/'+str(item)+"/"+str(p),target_size=(70,70))
            image = keras.preprocessing.image.img_to_array(image)
            image = image/255
            gender_train_x.append(image)
            gender_train_y.append(gender_dict[item])
        except:
            continue
#Converting lists to array
gender_train_x = np.asarray(gender_train_x)
gender_train_y = np.asarray(gender_train_y)
#Shuffling the dataset
gender_train_x, gender_train_y = shuffle(gender_train_x,gender_train_y)
#Categorical Splitting
gender_train_y=keras.utils.to_categorical(gender_train_y)


#===============================================ETHNICITY TRAINING MODEL DATA PREPROCESSING================================================================================


ethnicity_dict={
    'arab':0,
    'asian':1,
    'black':2,
    'hispanic':3,
    'indian':4,
    'white':5
}

ethnicity_train_x = []
ethnicity_train_y=[]

for item in os.listdir('./dataset/ethnicity'):
    for p in os.listdir('./dataset/ethnicity/'+str(item)):
        try:
            image = keras.preprocessing.image.load_img('./dataset/ethnicity/'+str(item)+"/"+str(p),target_size=(70,70))
            image = keras.preprocessing.image.img_to_array(image)
            image = image/255
            ethnicity_train_x.append(image)
            ethnicity_train_y.append(ethnicity_dict[item])
        except:
            continue
            
ethnicity_train_x = np.asarray(ethnicity_train_x)
ethnicity_train_y = np.asarray(ethnicity_train_y)

ethnicity_train_y=keras.utils.to_categorical(ethnicity_train_y)
ethnicity_train_x, ethnicity_train_y = shuffle(ethnicity_train_x,ethnicity_train_y)



#===============================================EMOTIONS TRAINING MODEL DATA PREPROCESSING====================================================================================



emotion_dict={
    'angry':0,
    'happy':1,
    'neutral':2,
    'sad':3
}
emotions_train_x = []
emotions_train_y=[]

for item in os.listdir('./dataset/emotions'):
    for p in os.listdir('./dataset/emotions/'+str(item)):
        try:
            image = keras.preprocessing.image.load_img('./dataset/emotions/'+str(item)+"/"+str(p),target_size=(70,70))
            image = keras.preprocessing.image.img_to_array(image)
            image = image/255
            emotions_train_x.append(image)
            emotions_train_y.append(emotion_dict[item])
        except:
            continue
emotions_train_x = np.asarray(emotions_train_x)
emotions_train_y = np.asarray(emotions_train_y)

emotions_train_y=keras.utils.to_categorical(emotions_train_y)

emotions_train_x, emotions_train_y= shuffle(emotions_train_x,emotions_train_y)




age_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

gender_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

emotions_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

ethnicity_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)



#==================================================AGE MODEL TRAINING=================================================================================



age_model = Sequential()

#Convolution layer 1
age_model.add(Conv2D(32,3,3,input_shape=(70,70,3),activation='relu'))
age_model.add(MaxPooling2D(pool_size=(2,2)))
#Convolution layer 2
age_model.add(Conv2D(64,3,3,activation='relu'))
age_model.add(MaxPooling2D(pool_size=(2,2)))
#Flattening 
age_model.add(Flatten())
#Feeding into fully Connected layers
age_model.add(Dense(128,activation='relu'))
age_model.add(Dense(64,activation='relu'))
#output layer
age_model.add(Dense(age_train_y.shape[1],activation='softmax')) 
age_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#age_model.fit(age_train_x,age_train_y,epochs=100,batch_size=32,validation_split=0.1)
age_generator.fit(age_train_x)
age_model.fit_generator(age_generator.flow(age_train_x, age_train_y, batch_size=32),steps_per_epoch=100, epochs=100)
age_model.save('age_model.h5')






#===============================================GENDER MODEL TRAINING====================================



gender_model = Sequential()

#Convolution layer 1
gender_model.add(Conv2D(32,3,3,input_shape=(70,70,3),activation='relu'))
gender_model.add(MaxPooling2D(pool_size=(2,2)))
#Convolution layer 2
gender_model.add(Conv2D(64,3,3,activation='relu'))
gender_model.add(MaxPooling2D(pool_size=(2,2)))
#Flattening 
gender_model.add(Flatten())
#Feeding into fully Connected layers
gender_model.add(Dense(128,activation='relu'))
gender_model.add(Dense(64,activation='relu'))
#output layer
gender_model.add(Dense(2,activation='softmax')) 
gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#gender_model.fit(gender_train_x,gender_train_y,epochs=100,batch_size=32,validation_split=0.1)
gender_generator.fit(age_train_x)
gender_model.fit_generator(gender_generator.flow(gender_train_x, gender_train_y, batch_size=32),steps_per_epoch=100, epochs=100)

gender_model.save('gender_model.h5')



#=================================================ETHNICITY MODEL TRAINING=================================


ethnicity_model = Sequential()

#Convolution layer 1
ethnicity_model.add(Conv2D(32,3,3,input_shape=(70,70,3),activation='relu'))
ethnicity_model.add(MaxPooling2D(pool_size=(2,2)))
#Convolution layer 2
ethnicity_model.add(Conv2D(64,3,3,activation='relu'))
ethnicity_model.add(MaxPooling2D(pool_size=(2,2)))
#Flattening 
ethnicity_model.add(Flatten())
#Feeding into fully Connected layers
ethnicity_model.add(Dense(128,activation='relu'))
ethnicity_model.add(Dense(64,activation='relu'))
#output layer
ethnicity_model.add(Dense(ethnicity_train_y.shape[1],activation='softmax')) 
ethnicity_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#ethnicity_model.fit(ethnicity_train_x,ethnicity_train_y,epochs=100,batch_size=32,validation_split=0.1)
ethnicity_generator.fit(age_train_x)
ethnicity_model.fit_generator(ethnicity_generator.flow(ethnicity_train_x, ethnicity_train_y, batch_size=32),steps_per_epoch=100, epochs=100)

ethnicity_model.save('ethnicity_model.h5')


#===========================================EMOTIONS MODEL TRAINING=======================================
emotions_model = Sequential()

#Convolution layer 1
emotions_model.add(Conv2D(32,3,3,input_shape=(70,70,3),activation='relu'))
emotions_model.add(MaxPooling2D(pool_size=(2,2)))
#Convolution layer 2
emotions_model.add(Conv2D(64,3,3,activation='relu'))
emotions_model.add(MaxPooling2D(pool_size=(2,2)))
#Flattening 
emotions_model.add(Flatten())
#Feeding into fully Connected layers
emotions_model.add(Dense(128,activation='relu'))
emotions_model.add(Dense(64,activation='relu'))
#output layer
emotions_model.add(Dense(emotions_train_y.shape[1],activation='softmax')) 
emotions_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#emotions_model.fit(emotions_train_x,emotions_train_y,epochs=100,batch_size=32,validation_split=0.1)
emotions_generator.fit(age_train_x)
emotions_model.fit_generator(emotions_generator.flow(emotions_train_x, emotions_train_y, batch_size=32),steps_per_epoch=100, epochs=100)

emotions_model.save('emotions_model.h5')



