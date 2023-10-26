# The SimCLR class implementation is based on the tutorial in https://keras.io/examples/vision/semisupervised_simclr/
import tensorflow as tf
import math
from scampi_unsupervised.data_augmentation import (
    random_crop,
    random_jitter,
    random_color_drop,
    ColorDrop,
    RandomBlur,
)

rng = tf.random.Generator.from_seed(42)


def get_projection_head(width, input_shape):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Dense(width, activation="relu"),
            tf.keras.layers.Dense(width),
            ],
            name="projection_head",
        )


def get_encoder(width, input_shape):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            tf.keras.layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            tf.keras.layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            tf.keras.layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(width, activation="relu"),
        ],
        name="encoder",
    )


class Augmenter(tf.keras.layers.Layer):
    def __init__(self, input_size, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.batch_size = batch_size

    def call(self, x, training=True):
        # rescale
        x = tf.cast(x, tf.float32) / 255.0
        if training:
            x = random_crop(
                x,
                min_area=0.25,
                height=self.input_size[0],
                width=self.input_size[1],
                batch_size=self.batch_size,
            )
            x = tf.image.random_flip_left_right(x)
            x = random_jitter(x, p=0.8, s=1.0)
            x = random_color_drop(x, p=0.2)
        return x


def get_augmenter(input_shape, min_area, brightness, jitter):
    zoom_factor = 1.0 - math.sqrt(min_area)
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Rescaling(1 / 255),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
            tf.keras.layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
            RandomColorAffine(brightness, jitter),
            #ColorDrop(p=0.2),
            RandomBlur(p=0.5, kernel_size=21),
        ]
    )



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


class ContrastiveModel(tf.keras.Model):
    """TensorFlow SimCLR model for contrastive learning."""

    def __init__(self, augmenter, encoder, projection_head, temperature=0.1, loss_implementation="simple", **kwargs):
        super().__init__(name="simclr", **kwargs)
        # === Hyperparameters ===
        self.temperature = temperature
        self.loss_implementation = loss_implementation
        # === Architecture ===
        self.augmenter = augmenter
        self.encoder = encoder
        self.projection_head = projection_head
        # === Linear probe ===
        self.linear_probe = None
        self.labeled_dataset = None

    def compile(self, optimizer, probe_optimizer, **kwargs):
        """
        Initialize the contrastive and probe optimizers and losses.
        Note that self.contrastive_loss is not here, but is defined as a method below.
        """
        super().compile(**kwargs)

        self.contrastive_optimizer = optimizer
        self.probe_optimizer = probe_optimizer

        self.probe_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )

        self.contrastive_loss_tracker = tf.keras.metrics.Mean(name="contrastive_loss")
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="contrastive_acc"
        )
        self.probe_loss_tracker = tf.keras.metrics.Mean(name="prediction_loss")
        self.probe_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="prediction_acc"
        )

    @property
    def metrics(self):
        return super().metrics + [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.probe_loss_tracker,
            self.probe_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        """
        Compute the contrastive loss - NT-Xent loss (normalized temperature-scaled cross entropy) - for two sets of projections.
        Possibly equal to the InfoNCE loss (information noise-contrastive estimation) described elsewhere.
        """
        
        batch_size = tf.shape(projections_1)[0]
        
        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        
        if self.loss_implementation == "simple":
            # Similaritites as in tutorial
            similarities1 = (
                tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
            )
            similarities2 = tf.transpose(similarities1)
        
        elif self.loss_implementation == "complicated":
            # Similarities as in the paper
            large_value = tf.eye(batch_size) * 1e9
            sim_12 = (
                tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
            )
            sim_21 = (
                tf.matmul(projections_2, projections_1, transpose_b=True) / self.temperature
            )
            sim_11 = (
                tf.matmul(projections_1, projections_1, transpose_b=True) / self.temperature
            ) - large_value
            sim_22 = (
                tf.matmul(projections_2, projections_2, transpose_b=True) / self.temperature
            ) - large_value
            similarities1 = tf.concat([sim_12, sim_11], axis=1)
            similarities2 = tf.concat([sim_21, sim_22], axis=1)
        else:
            raise ValueError("method must be 'simple' or 'complicated'")

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities1)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities2)
        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities1, from_logits=True
        )
        loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities2, from_logits=True
        )

        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, images):
        """
        In the Keras tutorial, both labeled and unlabeled images are concatenated for contrastive learning.
        Here, we only use the unlabeled images for contrastive learning.
        """
        augmented_images_1 = self.augmenter(images, training=True)
        augmented_images_2 = self.augmenter(images, training=True)

        with tf.GradientTape() as tape:
            # Each augmented image is passed through the encoder
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            # The contrastive loss is computed on the projections
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)

        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )

        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        #########################################
        ### The next part is the linear probe ###
        #########################################
        if self.linear_probe is not None:
            # Labels are only used in evalutation for an on-the-fly logistic regression
            images, labels = next(iter(self.labeled_dataset))
            images = images / 255.0
            # Sample weights are used for class balancing
            sample_weight = self.class_weights.lookup(labels)
            with tf.GradientTape() as tape:
                # the encoder is used in inference mode here to avoid regularization
                # and updating the batch normalization paramers if they are used
                features = self.encoder(images, training=False)
                class_logits = self.linear_probe(features, training=True)
                probe_loss = self.probe_loss(
                    labels, class_logits, sample_weight=sample_weight
                )
            gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
            self.probe_optimizer.apply_gradients(
                zip(gradients, self.linear_probe.trainable_weights)
            )
            self.probe_loss_tracker.update_state(probe_loss)
            self.probe_accuracy.update_state(labels, class_logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        labeled_images, labels = data

        # For testing the components are used with a training=False flag
        preprocessed_images = labeled_images / 255.0
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        # Only the probe metrics are logged at test time
        return {m.name: m.result() for m in self.metrics[2:]}
