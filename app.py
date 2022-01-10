import os
import logging
from io import BytesIO
from PIL import Image
from threading import Thread

#Prevents warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from flask_socketio import SocketIO, emit

import cv2
import base64
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from flask import Flask, make_response, render_template, copy_current_request_context

#Initializing variables
app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet', manage_session=False, cors_allowed_origins="*")
import eventlet
eventlet.monkey_patch()

frontalFaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model =  keras.models.load_model("./WearMask/Model/bestModel.h5")

finalSize = 128
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.9
thickness = 2
lineType = 1

def predict(frame):
    """ Obtaining prediction from CNN model """
    frame = tf.image.resize_with_pad(frame, finalSize, finalSize)
    frame = tf.expand_dims(frame, axis = 0)
    frame = frame / 255
    y_prob = np.around(model.predict(frame)[0] * 100, 1)
    return y_prob[0]

@socketio.on("image")
def genFrames(imageData):
    """ Video stream functionality """
    @copy_current_request_context
    def print_result():  
        idx = imageData.find('base64,')
        base64Data  = imageData[idx+7:]
        decoder = BytesIO()
        decoder.write(base64.b64decode(base64Data, ' /'))
        image = Image.open(decoder)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frontalFaces = frontalFaceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(128, 128),
                    flags = cv2.CASCADE_SCALE_IMAGE
        )          
        try:
            for (x, y, w, h) in frontalFaces:
                head = frame[y:y + h, x:x + w, :]
            y_prob = predict(head)
            for (x, y, w, h) in frontalFaces:
                if y_prob >= 50:
                    fontColor = (0,255,0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), fontColor, 5)
                    bottomLeftCornerOfText = (x, y - 10)
                    cv2.putText(frame, "%s %s" % ("Mask", y_prob), 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
                else:
                    fontColor = (0,0,255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), fontColor, 5)
                    bottomLeftCornerOfText = (x, y - 10)
                    cv2.putText(frame, "%s %s" % ("No Mask", np.around(100 - y_prob, 1)), 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
            ret, buffer = cv2.imencode('.jpg', frame)
            stringData = base64.b64encode(buffer).decode('utf-8')
            b64Head = 'data:image/jpeg;base64,'
            stringData = b64Head + stringData
            emit('response_back', {
                    "image": stringData, 
                })
        except:
            _, buffer = cv2.imencode('.jpg', frame)
            stringData = base64.b64encode(buffer).decode('utf-8')
            b64Head = 'data:image/jpeg;base64,'
            stringData = b64Head + stringData
            emit('response_back', {
                    "image": stringData, 
                })
    thread = Thread(target=print_result)
    thread.start()

@app.route("/")
def index():
    return make_response(render_template("index.html"))

@app.route("/app")
def appPage():
    return make_response(render_template("app.html"))

if __name__ == "__main__":
    socketio.run(app, debug = True)