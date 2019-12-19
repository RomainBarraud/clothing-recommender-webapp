import tensorflow as tf

model_face = tf.keras.models.load_model("gender_model.h5")
model_whole_body = tf.keras.models.load_model("model_whole.h5")
model_upper_body = tf.keras.models.load_model("model_upper.h5")
model_lower_body = tf.keras.models.load_model("model_lower.h5")
