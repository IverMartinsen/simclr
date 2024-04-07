import tensorflow as tf

model = tf.keras.applications.Xception(
    weights="imagenet",
    include_top=False, 
    pooling="avg", 
    input_shape=(224, 224, 3), 
    )

print(model.summary())