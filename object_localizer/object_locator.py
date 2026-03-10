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

    def __init__(self, input_shape=(200, 200, 3), num_of_output=8, steps_per_epoch=50,
                 objects_dir = 'objects', backgrounds_dir='backgrounds'):
        """Initialize ObjectLocator with additional parameters.

        Args:
            input_shape: Shape of input images
            num_of_output: Number of output values (bbox coords)
            steps_per_epoch: Steps per training epoch
            objects_dir: directory containing object images (.png)
            backgrounds_dir: directory containing background images (.jpg)
        """
        super().__init__(input_shape, num_of_output, steps_per_epoch)
        self.class_names = []  # To store class names corresponding to object images

        # Load all background images once to avoid repeated file I/O during training
        self.background_images = []
        for fname in sorted(os.listdir(backgrounds_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                bg_img = np.array(imageio.imread(os.path.join(backgrounds_dir, fname)))
                # Ensure 3-channel RGB
                if bg_img.ndim == 2:
                    bg_img = np.stack([bg_img] * 3, axis=-1)
                elif bg_img.shape[2] == 4:
                    bg_img = bg_img[:, :, :3]
                self.background_images.append(bg_img)
        print(f"Loaded {len(self.background_images)} background images from '{backgrounds_dir}'")

        # Load all object images once to avoid repeated file I/O during training
        self.object_images = []
        for fname in sorted(os.listdir(objects_dir)):
            if fname.lower().endswith(('.png',)):
                obj_img = np.array(imageio.imread(os.path.join(objects_dir, fname)))
                # append class name (without extension) to class_names list
                class_name = os.path.splitext(fname)[0]
                self.class_names.append(class_name)
                # Ensure 4-channel RGBA
                if obj_img.ndim == 2:
                    obj_img = np.stack([obj_img] * 4, axis=-1)
                elif obj_img.shape[2] == 3:
                    # Add alpha channel
                    obj_img = np.concatenate([obj_img, np.ones_like(obj_img[:, :, :1])], axis=-1)
                self.object_images.append(obj_img)
        print(f"Loaded {len(self.object_images)} object images from '{objects_dir}'")

    def build_model(self, multi_class=True, unfreeze_last_n_blocks=0):
        if multi_class:
            self.build_vgg16_backbone_multiclass_model(vgg_weights='imagenet', unfreeze_last_n_blocks=unfreeze_last_n_blocks)
        else:
            self.build_vgg16_backbone_model(vgg_weights='imagenet', output_activation_func='sigmoid')

    def _get_random_background_patch(self):
        """Select a random background image and crop a random patch of input_shape dimensions."""
        bg = self.background_images[np.random.randint(len(self.background_images))]
        bg_h, bg_w = bg.shape[0], bg.shape[1]

        # If background is smaller than needed, resize it up
        if bg_h < self.image_height or bg_w < self.image_width:
            scale = max(self.image_height / bg_h, self.image_width / bg_w)
            new_h = int(bg_h * scale) + 1
            new_w = int(bg_w * scale) + 1
            bg = (skimage_resize(bg, (new_h, new_w, 3), preserve_range=True)).astype(np.uint8)
            bg_h, bg_w = bg.shape[0], bg.shape[1]

        # Random crop
        row_start = np.random.randint(0, bg_h - self.image_height + 1)
        col_start = np.random.randint(0, bg_w - self.image_width + 1)

        return bg[row_start:row_start + self.image_height, col_start:col_start + self.image_width, :]

    def _create_random_location_for_actual_image(self):
        """Load a random object image and put it on a random location against a random background.
        Returns the image and grid-based targets (per-cell predictions, YOLO-style)."""

        # get the number of object images available and select one randomly
        actual_obj_img = None
        num_classes = len(self.object_images)
        if num_classes != 3:
            raise ValueError("Number of object images found do not match the expected number of classes.")
        else:
            # select a random object image from the preloaded list
            class_idx =  np.random.randint(num_classes)
            actual_obj_img = self.object_images[class_idx]

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

        # Get a random background patch as the canvas (instead of black)
        x = self._get_random_background_patch().astype(np.float64)

        # Grid cell dimensions in pixels
        cell_h = self.image_height / self.grid_h  # ~33.3 px for 200/6
        cell_w = self.image_width / self.grid_w

        # add random option whether the object will be placed in the image or not to create some negative samples (images with no objects)
        if np.random.rand() > 0.75:
            # Return the background patch as is, with no object — all cells empty
            x = x / 255.0  # Normalize to [0, 1]
            targets = {
                'bbox_output': np.zeros((self.num_cells, 4), dtype=np.float32),
                'class_output': np.zeros(self.num_cells, dtype=np.float32),
                'objectness_output': np.zeros((self.num_cells, 1), dtype=np.float32)
            }
            original_coordinates = np.array([0, 0, 0, 0])  # No object coordinates
            return x, targets, original_coordinates

        # else place the object in the image as usual

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

        # Use alpha channel to create a mask where the object exists
        obj_rgb = actual_obj[:, :, :3]            # RGB channels
        obj_alpha = actual_obj[:, :, 3]            # alpha channel (0 = transparent)
        mask = (obj_alpha > 0).astype(np.uint8)    # binary mask: 1 where object exists
        mask_3ch = mask[:, :, np.newaxis]           # (h, w, 1) for broadcasting across RGB

        # Multiply mask with background to zero out where object will go, then add object
        bg_patch = x[row0:row1, col0:col1, :]
        x[row0:row1, col0:col1, :] = bg_patch * (1 - mask_3ch) + obj_rgb * mask_3ch

        x = x / 255. # assuming image has 8 bit color (max 255)

        # === YOLO-style cell-relative target encoding ===
        # Find object center
        center_row = (row0 + row1) / 2.0
        center_col = (col0 + col1) / 2.0

        # Determine responsible cell (the cell whose region contains the object center)
        resp_cell_row = int(center_row / cell_h)
        resp_cell_col = int(center_col / cell_w)
        # Clamp to grid bounds (safety)
        resp_cell_row = min(resp_cell_row, self.grid_h - 1)
        resp_cell_col = min(resp_cell_col, self.grid_w - 1)
        resp_cell_idx = resp_cell_row * self.grid_w + resp_cell_col  # flat index 0..35

        # Cell-relative bbox: tx, ty are center offsets within the responsible cell [0,1]
        # tw, th are object size relative to full image [0,1]
        tx = (center_col - resp_cell_col * cell_w) / cell_w  # offset within cell [0,1]
        ty = (center_row - resp_cell_row * cell_h) / cell_h  # offset within cell [0,1]
        tw = (col1 - col0) / self.image_width                # width relative to image [0,1]
        th = (row1 - row0) / self.image_height                # height relative to image [0,1]

        # Build grid-shaped targets — all zeros except the responsible cell
        bbox_targets = np.zeros((self.num_cells, 4), dtype=np.float32)
        class_targets = np.zeros(self.num_cells, dtype=np.float32)
        obj_targets = np.zeros((self.num_cells, 1), dtype=np.float32)

        bbox_targets[resp_cell_idx] = [tx, ty, tw, th]
        class_targets[resp_cell_idx] = class_idx  # sparse class index (0, 1, or 2)
        obj_targets[resp_cell_idx] = 1.0           # object exists in this cell

        targets = {
            'bbox_output': bbox_targets,
            'class_output': class_targets,
            'objectness_output': obj_targets
        }

        return x, targets, original_coordinates

    def image_generator(self, batch_size=64):
        # generate image input and the grid-based targets to train randomly
        num_of_batches =  self.steps_per_epoch
        while True:
            # Each epoch will have num_of_batches
            for _ in range(num_of_batches):
                X = np.zeros((batch_size, *self.input_shape))
                Y_bbox = np.zeros((batch_size, self.num_cells, 4), dtype=np.float32)
                Y_class = np.zeros((batch_size, self.num_cells), dtype=np.float32)
                Y_obj = np.zeros((batch_size, self.num_cells, 1), dtype=np.float32)
                # Per-cell sample weights: 1.0 at responsible cell, 0.0 elsewhere
                # This masks bbox and class losses for cells with no object
                W_bbox = np.zeros((batch_size, self.num_cells), dtype=np.float32)
                W_class = np.zeros((batch_size, self.num_cells), dtype=np.float32)

                for i in range(batch_size):
                    x, targets, _ = self._create_random_location_for_actual_image()
                    X[i] = x
                    Y_bbox[i] = targets['bbox_output']
                    Y_class[i] = targets['class_output']
                    Y_obj[i] = targets['objectness_output']
                    # objectness_output is (num_cells, 1) — squeeze to (num_cells,) for sample weights
                    W_bbox[i] = targets['objectness_output'][:, 0]
                    W_class[i] = targets['objectness_output'][:, 0]

                yield (
                    X,
                    {'bbox_output': Y_bbox, 'class_output': Y_class, 'objectness_output': Y_obj},
                    {'bbox_output': W_bbox, 'class_output': W_class}
                )

    def train(self, batch_size=64, epochs=50, model_path='object_locator_model.keras'):
        """Train on real object data. Saves model after training."""
        if self.model is None:
            self.build_model()

        history = self.model.fit(
            self.image_generator(batch_size=batch_size),
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            verbose=1
        )

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
        """Predict bounding boxes on random images and draw ALL activated cell predictions.

        Without NMS, multiple overlapping boxes may appear around the object — this is
        expected and shows the raw per-cell output of the grid.

        Args:
            batch_size: Number of images to predict and visualize.
        """
        plt = get_plt()

        # Generate a batch of random images
        X = np.zeros((batch_size, *self.input_shape))
        all_coordinates = []

        for i in range(batch_size):
            x, _, coords = self._create_random_location_for_actual_image()
            X[i] = x
            all_coordinates.append(coords)

        # Predict the entire batch at once — returns list of 3 arrays:
        # [bbox_preds(batch,36,4), class_preds(batch,36,3), obj_preds(batch,36,1)]
        bbox_preds, class_preds, obj_preds = self.model.predict(X)

        # Grid cell dimensions in pixels
        cell_h = self.image_height / self.grid_h
        cell_w = self.image_width / self.grid_w

        # Color map for classes
        class_colors = ['r', 'g', 'b']  # one color per class

        # Visualize each result
        fig, axes = plt.subplots(1, batch_size, figsize=(5 * batch_size, 5))
        if batch_size == 1:
            axes = [axes]

        for i in range(batch_size):
            axes[i].imshow(X[i])
            original_coordinates = all_coordinates[i]
            detected_count = 0

            # Loop through all 36 cells — draw boxes for any cell with objectness > 0.5
            for cell_idx in range(self.num_cells):
                obj_score = obj_preds[i, cell_idx, 0]
                if obj_score <= 0.5:
                    continue

                detected_count += 1

                # Decode cell-relative bbox → image coordinates
                cell_row = cell_idx // self.grid_w
                cell_col = cell_idx % self.grid_w

                tx, ty, tw, th = bbox_preds[i, cell_idx]

                # Center of object in image coordinates
                center_col = (cell_col + tx) * cell_w
                center_row = (cell_row + ty) * cell_h
                obj_width = tw * self.image_width
                obj_height = th * self.image_height

                # Top-left corner for Rectangle (col, row)
                rect_col0 = center_col - obj_width / 2
                rect_row0 = center_row - obj_height / 2

                # Class prediction for this cell
                class_pred_idx = np.argmax(class_preds[i, cell_idx])
                class_pred_name = self.class_names[class_pred_idx]
                color = class_colors[class_pred_idx % len(class_colors)]

                # Draw box with transparency based on confidence
                alpha = float(np.clip(obj_score, 0.3, 1.0))
                rect = Rectangle(
                    (rect_col0, rect_row0), obj_width, obj_height,
                    linewidth=2, edgecolor=color, facecolor='none', alpha=alpha)
                axes[i].add_patch(rect)

                # Label with class name and confidence
                axes[i].text(rect_col0, rect_row0 - 2,
                           f"{class_pred_name} {obj_score:.2f}",
                           fontsize=6, color=color, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

                print(f"  Cell [{cell_row},{cell_col}] → {class_pred_name} "
                      f"(obj={obj_score:.3f}) bbox=({tx:.2f},{ty:.2f},{tw:.2f},{th:.2f})")

            print(f"\nImage {i+1} — {detected_count} cells activated (out of {self.num_cells})")
            print(f"  Ground truth (row0, col0, row1, col1): {original_coordinates}")

            title = f"Image {i+1}: {detected_count} detections"
            if detected_count == 0:
                title = f"Image {i+1} — No Object Detected"
            axes[i].set_title(title)

        plt.tight_layout()
        plt.show()
