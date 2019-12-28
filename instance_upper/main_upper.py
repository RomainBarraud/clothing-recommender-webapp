from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import os
import time
import numpy as np
import tensorflow as tf

from cv2 import cv2


model_clothing = tf.keras.models.load_model("./models/model_upper.h5")

CLOTHING_CATEGORY = ['Jacket', 'Sweater', 'Tee', 'Blazer']


def predict_clothing(image, IMG_HEIGHT, IMG_WIDTH):
    print("in predict upper")
    print(image)
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    #image = cv2.divide(image, scale=255)
    image = tf.cast(image, tf.float32)
    #image = tf.math.divide(image, 255.0)
    #image = image / 255.0
    print(image)
    image = np.expand_dims(image, 0)
    image = tf.cast(image, tf.float32)
    print("image shape:", image.shape)
    clothing = model_clothing.predict(image)
    clothing = CLOTHING_CATEGORY[np.argmax(clothing)]
    print("clothing: ", clothing)
    return clothing


app = Flask(__name__)


@app.route('/', methods=['POST'])
def predictor():
    if request.method == 'POST':
        data = request.json
        image = np.array(data['image'])
        image = image.astype('float32')
        IMG_HEIGHT = data['IMG_HEIGHT']
        IMG_WIDTH = data['IMG_WIDTH']
        print(jsonify(clothing=predict_clothing(image, IMG_HEIGHT, IMG_WIDTH)))

        return jsonify(clothing=predict_clothing(image, IMG_HEIGHT, IMG_WIDTH))


if __name__ == "__main__":
    app.debug = True
    app.run(host="127.0.0.1", port=5011)