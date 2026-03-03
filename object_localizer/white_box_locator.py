# white_box_locator.py - stage 1: detect white boxes on black background
from .base import Locator
from .imports import get_keras_layers, get_np, get_tf
from . import utils
import numpy as np

class WhiteBoxLocator(Locator):
    """Localize white rectangular boxes on black background."""

    def build_model(self):
        self.build_vgg16_backbone_model(vgg_weights='imagenet', output_activation_func='sigmoid')

    def image_generator(self, batch_size=64):
        # generete image input and the targets to train randomly
        num_of_batches =  self.steps_per_epoch
        while True:
            # Each epoch will have num_of_batches
            for _ in range(num_of_batches):
                X = np.zeros((batch_size, *self.input_shape))
                Y = np.zeros((batch_size, self.num_of_output))

                for i in range(batch_size):
                    # make the white boxes (normalized white = 1, black = 0)

                    #top-left corner
                    row0 = np.random.randint(90)
                    col0 = np.random.randint(90)

                    #bottom-right corner
                    row1 = np.random.randint(row0, 100)
                    col1 = np.random.randint(row0, 100)

                    X[i, row0:row1, col0:col1, :] = 1
                    
                    #normalize Y output (x1, y1, h, w)
                    Y[i, 0] = row0/100.
                    Y[i, 1] = col0/100.
                    Y[i, 2] = (row1 - row0)/100.
                    Y[i, 3] = (col1 - col0)/100.

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