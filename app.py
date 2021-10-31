import os
import logging
from io import BytesIO
from PIL import Image

#Prevents warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from flask_socketio import SocketIO, emit

import cv2
import base64
import numpy as np
from time import sleep
import tensorflow as tf
import tensorflow.keras as keras
from flask import Flask, make_response, render_template

#Initializing variables
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

frontalFaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model =  keras.models.load_model("./WearMask/Model/bestModel.h5")

def predict(frame):
    """ Obtaining prediction from CNN model """
    finalSize = 64
    frame = frame / 255
    frame = tf.image.resize_with_pad(frame, finalSize, finalSize)
    frame = tf.expand_dims(frame, axis = 0)
    frame = tf.image.rgb_to_grayscale(frame)
    y_prob = np.around(model.predict(frame)[0] * 100, 2)
    return y_prob[0]
"""
@app.before_first_request
def beforeFirstRequest():
    Thread(target=updateLoad).start()

def updateLoad():
    with app.app_context():
        while True:
            sleep(1)
            turbo.push(turbo.replace(render_template('probabilities.html'), 'load'))

@app.context_processor            
def genModel():
    try:
        frame= request.files.get('video')
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frontalFaces = frontalFaceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(128, 128),
                    flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(frontalFaces) != 0:
            for (x, y, w, h) in frontalFaces:
                frame = frame[y:y + h, x:x + w, :]
                y_prob = predict(frame)
        else:
            y_prob = -1 
    except:
        y_prob = -1
    return {"y_prob": y_prob}   
"""
@socketio.on("image")
def genFrames(imageData):
    """ Video stream functionality """
    sleep(0.1)
    try:
        idx = imageData.find('base64,')
        base64Data  = imageData[idx+7:]
        decoder = BytesIO()
        decoder.write(base64.b64decode(base64Data, ' /'))
        image = Image.open(decoder)
        frame = np.array(image)
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frontalFaces = frontalFaceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(128, 128),
                    flags = cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in frontalFaces:
            cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
            frame = frame[y - 25:y + h + 60, x +20 :x + w - 20, :]
            y_prob = predict(frame)
        ret, buffer = cv2.imencode('.jpeg', gray)
        stringData = base64.b64encode(buffer).decode('utf-8')
        b64Head = 'data:image/jpeg;base64,'
        stringData = b64Head + stringData

        if y_prob >=50:
            print(y_prob)
            emit('response_back', {
                "image": stringData, 
                "condition": "Mask", 
                "probability": str(y_prob)
            })
        else:
            emit('response_back', {
                "image": stringData, 
                "condition": "No Mask", 
                "probability": str(y_prob)
            })
    except:
        emit('response_back', {
                "image": stringData, 
                "condition": "Wait a moment", 
                "probability": "Calculating..."
            })

@app.route("/")
def index():
    return make_response(render_template("index.html"))

@app.route("/app")
def appPage():
    return make_response(render_template("app.html"))

if __name__ == "__main__":
    socketio.run(app, debug=True)