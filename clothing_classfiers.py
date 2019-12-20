import numpy as np
import tensorflow as tf

model_gender = tf.keras.models.load_model("./models/model_gender.h5")
model_whole_body = tf.keras.models.load_model("./models/model_whole.h5")
model_upper_body = tf.keras.models.load_model("./models/model_upper.h5")
model_lower_body = tf.keras.models.load_model("./models/model_lower.h5")

BODY_WHOLE = {}
BODY_UPPER = {}
BODY_LOWER = {}

def predict_gender(image):
    if (model_gender.predict(image) == 1):
        return "Female"
    else:
        return "Male"

def predict_upper(image, model_clothing, clothing_dict):
    clothing = clothing_dict[np.argmax(model_clothing.predict(image))]
    return clothing
