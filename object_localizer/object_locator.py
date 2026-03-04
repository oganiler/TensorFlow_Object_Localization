# object_locator.py - stage 2: locate actual objects in images
from .base import Locator
from .imports import get_keras_layers, get_np, get_tf, get_plt
from . import utils
import numpy as np
from matplotlib.patches import Rectangle

class ObjectLocator(Locator):
    """Localize actual objects in real images."""

    def build_model(self):
        self.build_vgg16_backbone_model(vgg_weights='imagenet', output_activation_func='sigmoid')

    def image_generator(self, batch_size=64):
        # TODO: load real images and bounding box annotations
        raise NotImplementedError("image_generator not yet implemented for ObjectLocator")

    def train(self, batch_size=64, epochs=50):
        """Train on real object data."""
        tf = get_tf()

        if self.model is None:
            self.build_model()

        ds = tf.data.Dataset.from_generator(
            lambda: self.image_generator(batch_size=batch_size),
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                (batch_size, *self.input_shape),
                (batch_size, self.num_of_output)
            )
        )

        history = self.model.fit(ds, steps_per_epoch=self.steps_per_epoch, epochs=epochs, verbose=1)
        return history

    def predict_and_visualize(self, image):
        """Predict bounding box on a real image and draw the result."""
        plt = get_plt()

        X = np.expand_dims(image, 0)
        p = self.model.predict(X)[0]

        h, w = image.shape[:2]
        print("Predicted (row0, col0, h, w):", p)

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        rect = Rectangle(
            (p[1]*w, p[0]*h),
            p[3]*w, p[2]*h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()
