import tensorflow as tf 
import keras.backend as kb
#kb._SYMBOLIC_SCOPE.value = True
import streamlit as st
from classify import prediction
from PIL import Image
import numpy as np
from keras.models import load_model
import time


#from tensorflow.keras.preprocessing.image import load_img
#from tensorflow.keras.preprocessing import image

# chargement du modele serialisé
model_path= "Models/model.h5"

model = load_model(model_path)
# on declar une liste avec les types de fichier pris en charge
file_types =['jpeg', 'jpg', 'png']
# ouverture et affichage de l'image d'entete 
image = Image.open('images/pneumonie.jpeg')
st.image(image, use_column_width=True)
st.title(" DETECTION DE PNEUMONIE ")

#section upload de l'image a classer
uploaded_file = st.file_uploader("Choisissez la radio...", type= file_types)
if uploaded_file is not None:
    Image = Image.open(uploaded_file)
    #img = Image.save('images/uploaded_file')
    #img = image.load_img(path, target_size = (150, 150))
    st.image(Image, caption=' Image choisie', use_column_width=True)
    st.write("")
    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete +1)
    
    message = prediction(uploaded_file, model)
    print(message)


add_selectbox = st.sidebar.selectbox(
    "Cette application permet le diagnostic d'une personne a partir d'une Radiographie pulmonaire afin de detecter la presence ou l'absence de pneumonie ",
    ("Single image", "Multiples images"),
    
)

