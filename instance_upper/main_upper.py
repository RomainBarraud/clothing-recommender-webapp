from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import os
import time
import numpy as np
import tensorflow as tf

from cv2 import cv2


IMG_HEIGHT = 256
IMG_WIDTH = 256

model_upper_body = tf.keras.models.load_model("./models/model_upper.h5")

UPPER_CATEGORY = ['Jacket', 'Sweater', 'Tee', 'Blazer']


def predict_upper(img_np, IMG_HEIGHT, IMG_WIDTH):
    print("in predict upper")
    image = cv2.imread(img_np)
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    print(image.shape)
    image = np.expand_dims(image, 0)
    image = tf.cast(image, tf.float32)
    print("image shape:", image.shape)
    clothing = model_upper_body.predict(image)
    clothing = UPPER_CATEGORY[np.argmax(clothing)]
    print("clothing: ", clothing)
    return clothing


app = Flask(__name__)


@app.route('/', methods=['POST'])
def predictor():
    if request.method == 'POST':
        data = request.json
        image = data['image']
        IMG_HEIGHT = data['IMG_HEIGHT']
        IMG_WIDTH = data['IMG_WIDTH']

        return predict_upper(image, IMG_HEIGHT, IMG_WIDTH)


if __name__ == "__main__":
    app.debug = True
    app.run(host="127.0.0.1", port=5011)


