import numpy as np
import tensorflow as tf
from cv2 import cv2

IMG_GENDER_HEIGHT = 150
IMG_GENDER_WIDTH = 150
IMG_HEIGHT = 256
IMG_WIDTH = 256

#model_gender = tf.keras.models.load_model("./models/model_gender.h5")
#model_whole_body = tf.keras.models.load_model("./models/model_whole.h5")
#model_upper_body = tf.keras.models.load_model("./models/model_upper.h5")
#model_lower_body = tf.keras.models.load_model("./models/model_lower.h5")

UPPER_CATEGORY = ['Jacket', 'Sweater', 'Tee', 'Blazer']
LOWER_CATEGORY = ['Jeans', 'Shorts', 'Skirt']
WHOLE_CATEGORY = ['Dress', 'Jumpsuit', 'Kimono']

def predict_gender(image, IMG_HEIGHT, IMG_WIDTH):
    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = np.expand_dims(image, -1)
    image = np.expand_dims(image, 0)
    image = tf.cast(image, tf.float32)
    prediction = model_gender.predict_proba(image)[0][0]
    if (prediction == 1):
        return "Female"
    else:
        return "Male"

def predict_upper(image, IMG_HEIGHT, IMG_WIDTH):
    print("in predict upper")
    image = cv2.resize(image, (IMG_GENDER_HEIGHT, IMG_GENDER_WIDTH))
    print(image.shape)
    image = np.expand_dims(image, 0)
    image = tf.cast(image, tf.float32)
    print("image shape:", image.shape)
    clothing = model_upper_body.predict(image)
    clothing = UPPER_CATEGORY[np.argmax(clothing)]
    print("clothing: ", clothing)
    return clothing

def predict_lower(image, IMG_HEIGHT, IMG_WIDTH):
    print("in predict lower")
    image = cv2.resize(image, (IMG_GENDER_HEIGHT, IMG_GENDER_WIDTH))
    print(image.shape)
    image = np.expand_dims(image, 0)
    image = tf.cast(image, tf.float32)
    print("image shape:", image.shape)
    clothing = model_lower_body.predict(image)
    clothing = LOWER_CATEGORY[np.argmax(clothing)]
    print("clothing: ", clothing)
    return clothing

def predict_whole(image, IMG_HEIGHT, IMG_WIDTH):
    print("in predict whole")
    image = cv2.resize(image, (IMG_GENDER_HEIGHT, IMG_GENDER_WIDTH))
    print(image.shape)
    #image = np.expand_dims(image, -1)
    image = np.expand_dims(image, 0)
    image = tf.cast(image, tf.float32)
    print("image shape:", image.shape)
    clothing = model_whole_body.predict(image)
    clothing = WHOLE_CATEGORY[np.argmax(clothing)]
    print("clothing: ", clothing)
    return clothing