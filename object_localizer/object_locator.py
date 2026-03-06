# object_locator.py - stage 2: locate actual objects in images
from .base import Locator
from .imports import get_keras_layers, get_np, get_tf, get_plt
from . import utils
import numpy as np
from matplotlib.patches import Rectangle
import imageio
import os
from skimage.transform import resize as skimage_resize

class ObjectLocator(Locator):
    """Localize actual objects in real images."""

    def __init__(self, input_shape=(200, 200, 3), num_of_output=4, steps_per_epoch=50, actual_image_path = 'charmander-tight.png'):
        """Initialize ObjectLocator with additional parameters.
        
        Args:
            input_shape: Shape of input images
            num_of_output: Number of output values (bbox coords)
            steps_per_epoch: Steps per training epoch
            actual_image_path: path of the image to be trained on different locations
        """
        super().__init__(input_shape, num_of_output, steps_per_epoch)
        self.actual_image_path = actual_image_path

    def build_model(self):
        self.build_vgg16_backbone_model(vgg_weights='imagenet', output_activation_func='sigmoid')

    def _create_random_location_for_actual_image(self):
        """Load and image and put it on a random location against black background.
        Returns the image and normalized targets (row0, col0, h, w)."""
        actual_obj_img = imageio.imread(self.actual_image_path)
        actual_obj = np.array(actual_obj_img)
        actual_obj_height, actual_obj_width, actual_obj_color = actual_obj_img.shape

        # Random scale factor to augment training data
        scale_factor = 0.5 + np.random.uniform(0.5, 1.5)
        actual_obj_height = int(actual_obj_height * scale_factor)
        actual_obj_width = int(actual_obj_width * scale_factor)

        #random flip for data augmentation
        if np.random.rand() < 0.5:
            actual_obj = np.fliplr(actual_obj)

        # Clamp to image bounds so the object always fits
        actual_obj_height = min(actual_obj_height, self.image_height)
        actual_obj_width = min(actual_obj_width, self.image_width)

        # Resize the actual image to match the new scaled dimensions
        actual_obj = (skimage_resize(actual_obj, (actual_obj_height, actual_obj_width, actual_obj_color),
                                     preserve_range=True)).astype(np.uint8)

        x = np.zeros(self.input_shape)

        #top-left corner
        row0 = np.random.randint(self.image_height - actual_obj_height)
        col0 = np.random.randint(self.image_width - actual_obj_width)

        #bottom-right corner
        row1 = row0 + actual_obj_height
        col1 = col0 + actual_obj_width

        original_coordinates  = np.array([
                row0, 
                col0, 
                row1, 
                col1
            ])

        # put the actual image and then normalize to 0-1 
        x[row0:row1, col0:col1, :] = actual_obj[:, :, :3]
        x = x / 255. # assuming image has 8 bit color (max 255)

        #normalized targets (row0, col0, h, w)
        targets = np.array([
            row0/self.image_height, 
            col0/self.image_width, 
            (row1 - row0)/self.image_height, 
            (col1 - col0)/self.image_width
            ], dtype=np.float64)

        return x, targets, original_coordinates

    def image_generator(self, batch_size=64):
        # generate image input and the targets to train randomly
        num_of_batches =  self.steps_per_epoch
        while True:
            # Each epoch will have num_of_batches
            for _ in range(num_of_batches):
                X = np.zeros((batch_size, *self.input_shape))
                Y = np.zeros((batch_size, self.num_of_output))

                for i in range(batch_size):
                    X[i], Y[i], _ = self._create_random_location_for_actual_image()

                yield X, Y

    def train(self, batch_size=64, epochs=50, model_path='object_locator_model.keras'):
        """Train on real object data. Saves model after training."""
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

        self.model.save(model_path)
        print(f"Model saved to {model_path}")

        return history

    def load_model(self, model_path='object_locator_model.keras'):
        """Load a previously saved model, skipping training."""
        tf = get_tf()
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"No saved model found at {model_path}")
            return False

    def predict_and_visualize(self, batch_size=1):
        """Predict bounding boxes on random images and draw the results.

        Args:
            batch_size: Number of images to predict and visualize.
        """
        plt = get_plt()

        # Generate a batch of random images
        X = np.zeros((batch_size, *self.input_shape))
        Y = np.zeros((batch_size, self.num_of_output))
        all_coordinates = []

        for i in range(batch_size):
            X[i], Y[i], coords = self._create_random_location_for_actual_image()
            all_coordinates.append(coords)

        # Predict the entire batch at once
        predictions = self.model.predict(X)

        # Visualize each result
        fig, axes = plt.subplots(1, batch_size, figsize=(5 * batch_size, 5))
        if batch_size == 1:
            axes = [axes]

        for i in range(batch_size):
            p = predictions[i]
            original_coordinates = all_coordinates[i]

            row0 = int(p[0]*self.image_height)
            col0 = int(p[1]*self.image_width)
            row1 = int(row0 + p[2]*self.image_height)
            col1 = int(col0 + p[3]*self.image_width)

            print(f"\n--- Image {i+1} ---")
            print("Ground truth (row0, col0, row1, col1):", original_coordinates)
            #make an array of predicted coordinates to match the format of original_coordinates for easier comparison
            predicted_coordinates = np.array([row0, col0, row1, col1])
            print("Predicted    (row0, col0, row1, col1):", predicted_coordinates)

            axes[i].imshow(X[i])
            rect = Rectangle(
                (p[1]*self.image_width, p[0]*self.image_height),
                p[3]*self.image_width, p[2]*self.image_height, linewidth=1, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].set_title(f"Image {i+1}")

        plt.tight_layout()
        plt.show()
