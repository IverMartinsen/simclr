import tensorflow as tf

rng = tf.random.Generator.from_seed(42)

@tf.function
def random_crop(
    images,
    batch_size,
    height,
    width,
    min_area=0.08,
    min_aspect_ratio=3 / 4,
    max_aspect_ratio=4 / 3,
):
    """Randomly crop images. Default values as in SimCLR and BYOL papers."""
    # sample random area
    img_area = height * width
    target_area = rng.uniform([], min_area, 1.0, dtype=tf.float32) * img_area
    # sample random aspect ratio
    log_ratio = (tf.math.log(min_aspect_ratio), tf.math.log(max_aspect_ratio))
    aspect_ratio = tf.math.exp(rng.uniform([], *log_ratio, dtype=tf.float32))
    # compute crop dimensions
    w = tf.cast(tf.round(tf.sqrt(target_area * aspect_ratio)), tf.int32)
    h = tf.cast(tf.round(tf.sqrt(target_area / aspect_ratio)), tf.int32)
    # clip crop dimensions
    w = tf.minimum(w, width)
    h = tf.minimum(h, height)
    # apply random crop
    images = tf.image.random_crop(images, (batch_size, h, w, 3))
    # resize to original size
    return tf.image.resize(images, (height, width))


class RandomCrop(tf.keras.layers.Layer):
    def __init__(
        self,
        height,
        width,
        min_area=0.08,
        min_aspect_ratio=3 / 4,
        max_aspect_ratio=4 / 3,
        seed=0,
        **kwargs,
    ):
        super(RandomCrop, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.min_area = min_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def call(self, images, training=None):
        if training:
            return random_crop(
                images,
                tf.shape(images)[0],
                self.height,
                self.width,
                self.min_area,
                self.min_aspect_ratio,
                self.max_aspect_ratio,
            )
        return images

    def get_config(self):
        config = super(RandomCrop, self).get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "min_area": self.min_area,
                "min_aspect_ratio": self.min_aspect_ratio,
                "max_aspect_ratio": self.max_aspect_ratio,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_signature(self, input_signature):
        return input_signature
