import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import cv2
import numpy as np
from time import sleep
import tensorflow as tf
from threading import Thread
from turbo_flask import Turbo
import tensorflow.keras as keras
from flask import Flask, Response, make_response, render_template, request




app = Flask(__name__)
turbo = Turbo(app)
camera = cv2.VideoCapture(0)
frontalFaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model =  keras.models.load_model("./UseMask/Model/bestModel.h5")

def predict(frame):
    finalSize = 64
    frame = frame / 255
    frame = tf.image.resize_with_pad(frame, finalSize, finalSize)
    frame = tf.expand_dims(frame, axis = 0)
    frame = tf.image.rgb_to_grayscale(frame)
    y_prob = np.around(model.predict(frame)[0] * 100, 2)
    return y_prob[0]

@app.before_first_request
def before_first_request():
    Thread(target=update_load).start()

def update_load():
    with app.app_context():
        while True:
            sleep(1)
            turbo.push(turbo.replace(render_template('probabilities.html'), 'load'))

@app.context_processor            
def genModel():
    try:
        success, frame = camera.read()  # read the camera frame
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
    
def genFrames():
    while True:
        try:
            success, frame = camera.read()  # read the camera frame
            frame = cv2.flip(frame,1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frontalFaces = frontalFaceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(128, 128),
                        flags = cv2.CASCADE_SCALE_IMAGE
            )

            if not success:
                break
            else:
                for (x, y, w, h) in frontalFaces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                #    frame = frame[y:y + h, x + 10:x + w - 10, :]
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except:
            pass


@app.route("/")
def index():
    return make_response(render_template("index.html"))

@app.route("/app")
def appPage():
    return make_response(render_template("app.html"))

@app.route('/video_feed')
def video_feed():
    return Response(genFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug = True)