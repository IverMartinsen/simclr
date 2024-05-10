import tensorflow as tf


rng = tf.random.Generator.from_seed(42)


class RandomCrop(tf.keras.layers.Layer):
    def __init__(
        self,
        height,
        width,
        min_area=0.08,
        min_aspect_ratio=3 / 4,
        max_aspect_ratio=4 / 3,
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


class RandomColorAffine(tf.keras.layers.Layer):
    """
    Random color affine transformations as in Keras tutorial
    """

    def __init__(self, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.jitter = jitter

    def get_config(self):
        config = super().get_config()
        config.update({"brightness": self.brightness, "jitter": self.jitter})
        return config

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]
            # Same for all colors
            brightness_scales = 1 + rng.uniform(
                (batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness
            )
            # Different for all colors
            jitter_matrices = rng.uniform(
                (batch_size, 1, 3, 3), minval=-self.jitter, maxval=self.jitter
            )
            # Combine brightness and jitter
            color_transforms = (
                tf.eye(3, batch_shape=[batch_size, 1]) * brightness_scales
                + jitter_matrices
            )
            # cast to input type
            color_transforms = tf.cast(color_transforms, images.dtype)
            # Apply all color transformations
            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)
        return images


class ColorDrop(tf.keras.layers.Layer):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    
    def call(self, x, training=None):
        if training:
            x = random_color_drop(x, self.p)
        return x


@tf.function
def random_color_drop(images, p=0.2):
    """Randomly convert images to grayscale with probability p=0.2."""
    images = tf.cond(
        rng.uniform(shape=[]) < p, lambda: color_drop(images), lambda: images
    )
    return images

@tf.function
def color_drop(x):
    """Convert to grayscale. Shape is preserved."""
    rgb_weights = [0.2989, 0.5870, 0.1140]
    x = x * tf.reshape(tf.constant(rgb_weights), [1, 1, 1, 3])
    x = tf.reduce_sum(x, axis=3, keepdims=True)
    x = tf.tile(x, [1, 1, 1, 3])
    return x


class RandomBlur(tf.keras.layers.Layer):
    def __init__(self, p=0.5, kernel_size=9, min_sigma=0.1, max_sigma=2.0):
        super().__init__()
        self.p = p
        self.kernel_size = kernel_size
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
    
    def call(self, x, training=None):
        if training:
            tf.cond(
                rng.uniform(shape=[]) < self.p,
                lambda: random_blur(x, self.kernel_size, self.min_sigma, self.max_sigma),
                lambda: x,
            )
        return x


@tf.function
def random_blur(images, kernel_size=9, min_sigma=0.1, max_sigma=2.0):
    """Apply Gaussian with random sigma."""
    sigma = rng.uniform([], min_sigma, max_sigma, dtype=tf.float32)
    images = gaussian_filter_2d(images, kernel_size=kernel_size, sigma=sigma)
    return images

@tf.function
def gaussian_filter_2d(images, kernel_size, sigma):
    """Apply Gaussian filter to images."""
    channels = 3
    gaussian_kernel_2d = _get_gaussian_kernel_2d(sigma, kernel_size)
    gaussian_kernel_2d = tf.reshape(
        gaussian_kernel_2d, [kernel_size, kernel_size, 1, 1]
    )
    gaussian_kernel_2d = tf.tile(gaussian_kernel_2d, [1, 1, channels, 1])

    images = tf.nn.depthwise_conv2d(
        images, gaussian_kernel_2d, strides=[1, 1, 1, 1], padding="SAME"
    )
    return tf.clip_by_value(images, 0, 1)

@tf.function
def _get_gaussian_kernel_2d(sigma, filter_shape):
    """Compute 2D Gaussian kernel."""
    sigma = tf.convert_to_tensor(sigma)
    x = tf.range(-filter_shape // 2 + 1, filter_shape // 2 + 1)
    x = tf.cast(x**2, sigma.dtype)
    x = tf.nn.softmax(-x / (2.0 * (sigma**2)))

    return tf.matmul(x[:, tf.newaxis], x[tf.newaxis, :])
