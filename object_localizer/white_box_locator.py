# white_box_locator.py - stage 1: detect white boxes on black background
from .base import Locator
from .imports import get_keras_layers, get_np, get_tf, get_plt
from . import utils
import numpy as np
from matplotlib.patches import Rectangle

class WhiteBoxLocator(Locator):
    """Localize white rectangular boxes on black background."""

    def build_model(self):
        self.build_vgg16_backbone_model(vgg_weights='imagenet', output_activation_func='sigmoid')

    def _create_random_box_image(self):
        """Create a single random white box on black background.
        Returns the image and normalized targets (row0, col0, h, w)."""
        x = np.zeros(self.input_shape)

        #top-left corner
        row0 = np.random.randint(90)
        col0 = np.random.randint(90)

        #bottom-right corner
        row1 = np.random.randint(row0, 100)
        col1 = np.random.randint(col0, 100)

        x[row0:row1, col0:col1, :] = 1

        #normalized targets (row0, col0, h, w)
        targets = np.array([row0/100., col0/100., (row1 - row0)/100., (col1 - col0)/100.])
        return x, targets

    def image_generator(self, batch_size=64):
        # generate image input and the targets to train randomly
        num_of_batches =  self.steps_per_epoch
        while True:
            # Each epoch will have num_of_batches
            for _ in range(num_of_batches):
                X = np.zeros((batch_size, *self.input_shape))
                Y = np.zeros((batch_size, self.num_of_output))

                for i in range(batch_size):
                    X[i], Y[i] = self._create_random_box_image()

                yield X, Y
       
    def train(self, batch_size=64, epochs=50):
        """Train on synthetic white box data using image_generator."""
        tf = get_tf()
        
        if self.model is None:
            self.build_model()
        
        # Create dataset from your image_generator
        ds = tf.data.Dataset.from_generator(
            lambda: self.image_generator(batch_size=batch_size),
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                (batch_size, *self.input_shape),
                (batch_size, self.num_of_output)
            )
        )
        
        self.model.fit(ds, steps_per_epoch=self.steps_per_epoch, epochs=epochs, verbose=1)

    def predict_and_visualize(self):
        """Generate a random image, predict bounding box, and draw the result."""
        plt = get_plt()

        # Generate a random image
        x, targets = self._create_random_box_image()
        print("Ground truth (row0, col0, h, w):", targets)

        # Predict
        X = np.expand_dims(x, 0)
        p = self.model.predict(X)[0]
        print("Predicted    (row0, col0, h, w):", p)

        # Draw the box
        fig, ax = plt.subplots(1)
        ax.imshow(x)
        rect = Rectangle(
            (p[1]*100, p[0]*100),
            p[3]*100, p[2]*100, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()