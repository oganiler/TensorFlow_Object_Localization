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

    @staticmethod
    def _iou_wh(w1, h1, w2, h2):
        """Compute IoU between two boxes centered at the same point, given only width/height.
        Used for anchor matching — position doesn't matter, only shape overlap."""
        inter_w = min(w1, w2)
        inter_h = min(h1, h2)
        inter = inter_w * inter_h
        union = w1 * h1 + w2 * h2 - inter
        return inter / (union + 1e-6)

    def _slot_to_scale(self, slot_idx):
        """Map a flat slot index to its scale dict and local index within that scale.

        The flat index space is partitioned: [block3 slots | block4 slots | block5 slots].
        Returns (scale_dict, local_idx) where local_idx is relative to that scale's offset.
        """
        for scale in self.scales:
            if slot_idx < scale['offset'] + scale['num_slots']:
                return scale, slot_idx - scale['offset']
        # Fallback (should never reach here)
        last = self.scales[-1]
        return last, slot_idx - last['offset']

    def _create_random_location_for_actual_image(self):
        """Load a random object image and put it on a random location against a random background.
        Returns the image and anchor-based grid targets (per-cell, per-anchor, YOLO-style)."""

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

        # add random option whether the object will be placed in the image or not to create some negative samples (images with no objects)
        if np.random.rand() > 0.75:
            # Return the background patch as is, with no object — all anchor slots empty
            x = x / 255.0  # Normalize to [0, 1]
            targets = {
                'bbox_output': np.zeros((self.total_anchors, 4), dtype=np.float32),
                'class_output': np.zeros(self.total_anchors, dtype=np.float32),
                'objectness_output': np.zeros((self.total_anchors, 1), dtype=np.float32)
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

        # === Multi-scale cell-relative target encoding with ANCHOR MATCHING ===
        # Find object center and normalized dimensions
        center_row = (row0 + row1) / 2.0
        center_col = (col0 + col1) / 2.0
        obj_w = (col1 - col0) / self.image_width   # width relative to image [0,1]
        obj_h = (row1 - row0) / self.image_height   # height relative to image [0,1]

        # Multi-scale anchor matching: find the best (scale, anchor) pair
        # by comparing object shape against each scale's anchor shapes via IoU_wh.
        # This determines WHICH feature map level detects this object.
        best_scale_idx = 0
        best_anchor_idx = 0
        best_iou = 0.0
        for s_idx, scale in enumerate(self.scales):
            for a_idx, (aw, ah) in enumerate(scale['anchors']):
                iou = self._iou_wh(obj_w, obj_h, aw, ah)
                if iou > best_iou:
                    best_iou = iou
                    best_scale_idx = s_idx
                    best_anchor_idx = a_idx

        # Compute responsible cell within the MATCHED scale's grid
        scale = self.scales[best_scale_idx]
        cell_h = self.image_height / scale['grid_h']
        cell_w = self.image_width / scale['grid_w']
        resp_cell_row = min(int(center_row / cell_h), scale['grid_h'] - 1)
        resp_cell_col = min(int(center_col / cell_w), scale['grid_w'] - 1)
        resp_cell_idx = resp_cell_row * scale['grid_w'] + resp_cell_col

        # Cell-relative bbox: tx, ty are center offsets within the responsible cell [0,1]
        tx = (center_col - resp_cell_col * cell_w) / cell_w
        ty = (center_row - resp_cell_row * cell_h) / cell_h

        # Flat index using scale offset: [block3 slots | block4 slots | block5 slots]
        flat_idx = scale['offset'] + resp_cell_idx * scale['num_anchors'] + best_anchor_idx

        # Build targets — all zeros except the matched anchor slot
        bbox_targets = np.zeros((self.total_anchors, 4), dtype=np.float32)
        class_targets = np.zeros(self.total_anchors, dtype=np.float32)
        obj_targets = np.zeros((self.total_anchors, 1), dtype=np.float32)

        bbox_targets[flat_idx] = [tx, ty, obj_w, obj_h]
        class_targets[flat_idx] = class_idx  # sparse class index (0, 1, or 2)
        obj_targets[flat_idx] = 1.0           # object exists at this anchor slot

        targets = {
            'bbox_output': bbox_targets,
            'class_output': class_targets,
            'objectness_output': obj_targets
        }

        return x, targets, original_coordinates

    def image_generator(self, batch_size=64):
        # generate image input and anchor-based grid targets to train randomly
        num_of_batches =  self.steps_per_epoch
        while True:
            # Each epoch will have num_of_batches
            for _ in range(num_of_batches):
                X = np.zeros((batch_size, *self.input_shape))
                Y_bbox = np.zeros((batch_size, self.total_anchors, 4), dtype=np.float32)
                Y_class = np.zeros((batch_size, self.total_anchors), dtype=np.float32)
                Y_obj = np.zeros((batch_size, self.total_anchors, 1), dtype=np.float32)
                # Per-slot sample weights: 1.0 at matched anchor slot, 0.0 elsewhere
                # This masks bbox and class losses for empty anchor slots
                W_bbox = np.zeros((batch_size, self.total_anchors), dtype=np.float32)
                W_class = np.zeros((batch_size, self.total_anchors), dtype=np.float32)
                # Objectness weight: positive slots get 1.0, negative slots get noobj_weight
                # This rebalances the 107:1 neg:pos ratio to prevent the model
                # from learning "always predict low objectness"
                W_obj = np.full((batch_size, self.total_anchors), self.noobj_weight, dtype=np.float32)

                for i in range(batch_size):
                    x, targets, _ = self._create_random_location_for_actual_image()
                    X[i] = x
                    Y_bbox[i] = targets['bbox_output']
                    Y_class[i] = targets['class_output']
                    Y_obj[i] = targets['objectness_output']
                    # objectness_output is (total_anchors, 1) — squeeze for sample weights
                    obj_mask = targets['objectness_output'][:, 0]  # 1.0 at matched slot, 0.0 elsewhere
                    W_bbox[i] = obj_mask
                    W_class[i] = obj_mask
                    # Positive objectness slots get full weight (1.0),
                    # negatives keep the noobj_weight baseline set above
                    W_obj[i] = np.where(obj_mask > 0.5, 1.0, self.noobj_weight)

                # Keras 3 with TF backend: use tuples for multi-output targets/weights
                # Order must match Model(outputs=[bbox_output, class_output, objectness_output])
                yield (
                    X,
                    (Y_bbox, Y_class, Y_obj),
                    (W_bbox, W_class, W_obj)
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

    @staticmethod
    def _compute_iou_boxes(box1, box2):
        """Compute IoU between two boxes in (col0, row0, col1, row1) format."""
        inter_col0 = max(box1[0], box2[0])
        inter_row0 = max(box1[1], box2[1])
        inter_col1 = min(box1[2], box2[2])
        inter_row1 = min(box1[3], box2[3])
        inter_area = max(0, inter_col1 - inter_col0) * max(0, inter_row1 - inter_row0)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area / (area1 + area2 - inter_area + 1e-6)

    @staticmethod
    def _non_max_suppression(detections, iou_threshold=0.5):
        """Greedy NMS: suppress overlapping boxes, keeping the highest-confidence ones.

        Args:
            detections: list of dicts with keys 'box' (col0,row0,col1,row1), 'score',
                        'class_name', 'color', 'anchor_idx', 'cell_row', 'cell_col',
                        'width', 'height', 'rect_col0', 'rect_row0'
            iou_threshold: IoU threshold above which a lower-confidence box is suppressed.

        Returns:
            Filtered list of detections (survivors only).
        """
        if len(detections) == 0:
            return []

        # Sort by objectness score descending
        detections = sorted(detections, key=lambda d: d['score'], reverse=True)
        keep = []

        while detections:
            best = detections.pop(0)
            keep.append(best)
            # Remove any remaining detection that overlaps too much with the kept one
            detections = [
                d for d in detections
                if ObjectLocator._compute_iou_boxes(best['box'], d['box']) < iou_threshold
            ]

        return keep

    def predict_and_visualize(self, batch_size=1):
        """Predict bounding boxes with multi-scale anchors, apply NMS, and visualize.

        Each scale's feature map produces predictions at its own grid resolution.
        NMS filters duplicate/overlapping boxes across all scales.

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
        # [bbox_preds(batch,805,4), class_preds(batch,805,3), obj_preds(batch,805,1)]
        bbox_preds, class_preds, obj_preds = self.model.predict(X)

        # Color map for classes
        class_colors = ['r', 'g', 'b']  # one color per class

        # Visualize each result
        fig, axes = plt.subplots(1, batch_size, figsize=(5 * batch_size, 5))
        if batch_size == 1:
            axes = [axes]

        for i in range(batch_size):
            axes[i].imshow(X[i])
            original_coordinates = all_coordinates[i]

            # === Diagnostics: objectness score distribution for this image ===
            all_obj_scores = obj_preds[i, :, 0]
            top5_indices = np.argsort(all_obj_scores)[-5:][::-1]
            print(f"\nImage {i+1} — Top 5 objectness scores:")
            for rank, idx in enumerate(top5_indices):
                scale_d, local_d = self._slot_to_scale(idx)
                cell_idx_d = local_d // scale_d['num_anchors']
                anchor_idx_d = local_d % scale_d['num_anchors']
                print(f"  #{rank+1}: slot {idx} ({scale_d['name']}, cell {cell_idx_d}, "
                      f"anchor {anchor_idx_d}) -> obj={all_obj_scores[idx]:.4f}")

            # === Phase 1: Collect all raw detections above threshold ===
            obj_threshold = 0.6  # above sigmoid midpoint "uncertainty zone" (~0.5)
            raw_detections = []
            for slot_idx in range(self.total_anchors):
                obj_score = obj_preds[i, slot_idx, 0]
                if obj_score <= obj_threshold:
                    continue

                # Scale-aware decoding: determine which feature map produced this slot
                scale, local_idx = self._slot_to_scale(slot_idx)
                cell_idx = local_idx // scale['num_anchors']
                anchor_idx = local_idx % scale['num_anchors']
                cell_row = cell_idx // scale['grid_w']
                cell_col = cell_idx % scale['grid_w']

                # Cell dimensions for THIS scale's grid
                cell_h = self.image_height / scale['grid_h']
                cell_w = self.image_width / scale['grid_w']

                tx, ty, tw, th = bbox_preds[i, slot_idx]

                # Decode cell-relative bbox -> image coordinates
                center_col = (cell_col + tx) * cell_w
                center_row = (cell_row + ty) * cell_h
                obj_width = tw * self.image_width
                obj_height = th * self.image_height

                # Corner coordinates for NMS IoU computation
                rect_col0 = center_col - obj_width / 2
                rect_row0 = center_row - obj_height / 2
                rect_col1 = rect_col0 + obj_width
                rect_row1 = rect_row0 + obj_height

                # Class prediction for this anchor slot
                class_pred_idx = np.argmax(class_preds[i, slot_idx])
                class_pred_name = self.class_names[class_pred_idx]
                color = class_colors[class_pred_idx % len(class_colors)]

                raw_detections.append({
                    'box': (rect_col0, rect_row0, rect_col1, rect_row1),
                    'score': float(obj_score),
                    'class_name': class_pred_name,
                    'color': color,
                    'anchor_idx': anchor_idx,
                    'cell_row': cell_row,
                    'cell_col': cell_col,
                    'width': obj_width,
                    'height': obj_height,
                    'rect_col0': rect_col0,
                    'rect_row0': rect_row0,
                    'scale_name': scale['name'],
                })

            # === Phase 2: Apply Non-Max Suppression (cross-scale) ===
            kept = self._non_max_suppression(raw_detections, iou_threshold=0.3)

            print(f"\nImage {i+1} — {len(raw_detections)} raw detections -> "
                  f"{len(kept)} after NMS (out of {self.total_anchors} anchor slots)")
            print(f"  Ground truth (row0, col0, row1, col1): {original_coordinates}")

            # === Phase 3: Draw surviving boxes ===
            for d in kept:
                alpha = float(np.clip(d['score'], 0.3, 1.0))
                rect = Rectangle(
                    (d['rect_col0'], d['rect_row0']), d['width'], d['height'],
                    linewidth=2, edgecolor=d['color'], facecolor='none', alpha=alpha)
                axes[i].add_patch(rect)

                # Label with class name, confidence, scale, and anchor index
                axes[i].text(d['rect_col0'], d['rect_row0'] - 2,
                           f"{d['class_name']} {d['score']:.2f} ({d['scale_name']})",
                           fontsize=6, color=d['color'], fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

                print(f"  {d['scale_name']} cell [{d['cell_row']},{d['cell_col']}] "
                      f"anchor {d['anchor_idx']} -> "
                      f"{d['class_name']} (obj={d['score']:.3f}) "
                      f"box {d['width']:.0f}×{d['height']:.0f}px")

            title = f"Image {i+1}: {len(kept)} detections"
            if len(kept) == 0:
                title = f"Image {i+1} — No Object Detected"
            axes[i].set_title(title)

        plt.tight_layout()
        plt.show()
