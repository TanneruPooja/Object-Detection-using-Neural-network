import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np 

st.header('Vehicle Type Detection Model')
model = load_model(r'C:\Users\tanne\Downloads\Group Project\Group Project\Image_classify.keras')


#Adding the list of vehicle types in our dataset
data_cat = ['bus', 'car', 'motorcycle', 'truck']
img_height = 180
img_width = 180
image = st.text_input('Enter Image name', r'C:\Users\tanne\Downloads\Group Project\Group Project\bus.jpg')


image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)

#Printing results
st.image(image, width=200)
st.write('Vehicle Type in image is ' + data_cat[np.argmax(score)])
st.write('With accuracy of ' + str(np.max(score)*100))