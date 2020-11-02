import numpy as np
import tensorflow as tf 
import os 
from tensorflow import keras 
import pandas as pd
import cv2 
import matplotlib.pyplot as plt
import streamlit as st

def prediction(image, model):
    img = plt.imread(image)
    img = cv2.resize(img, (150, 150))
    img = np.dstack([img, img, img])
    img = img.astype('float32')/ 255
    img = np.array(img)
    img = img.reshape([1]+ list(img.shape))
    preds = model.predict(img)
    if preds[0][0]>0.7:
        message = st.error("pneumonie présente")
    else :
        message = st.success("Radio normale")


    return message
