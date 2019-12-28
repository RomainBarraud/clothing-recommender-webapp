from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import os
import time
import numpy as np
import tensorflow as tf

from cv2 import cv2


model_gender = tf.keras.models.load_model("./models/model_gender.h5")


def predict_gender(image, IMG_HEIGHT, IMG_WIDTH):
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = np.expand_dims(image, -1)
    image = np.expand_dims(image, 0)
    image = tf.cast(image, tf.float32)
    prediction = model_gender.predict_proba(image)[0][0]
    print("gender proba: ", prediction)
    if (prediction == 1):
        return "Female"
    else:
        return "Male"


app = Flask(__name__)


@app.route('/', methods=['POST'])
def predictor():
    if request.method == 'POST':
        print("in gender instance")
        data = request.json
        image = np.array(data['image'])
        image = image.astype('float32')
        IMG_HEIGHT = data['IMG_HEIGHT']
        IMG_WIDTH = data['IMG_WIDTH']
        print("will print gender jsonify")
        print(jsonify(clothing=predict_gender(image, IMG_HEIGHT, IMG_WIDTH)))
        print("printed gender jsonify")
        return jsonify(clothing=predict_gender(image, IMG_HEIGHT, IMG_WIDTH))


if __name__ == "__main__":
    app.debug = True
    app.run(host="127.0.0.1", port=5041)