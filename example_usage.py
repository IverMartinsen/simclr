import tensorflow as tf


encoder = tf.keras.applications.Xception(
    weights="path/to/weights",
    include_top=False, 
    pooling="avg", 
    input_shape=(224, 224, 3),
)

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'path/to/dataset',
    image_size=(224, 224),
    shuffle=False,
)

# need to rescale the images
dataset = dataset.map(lambda x, y: (x / 255.0, y))

features = encoder.predict(dataset)
