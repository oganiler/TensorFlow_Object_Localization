# base.py - abstract base class with common logic
from abc import ABC, abstractmethod
from .imports import get_tf, get_vgg16, get_keras_layers, get_binary_crossentropy

class Locator(ABC):
    """Base class for object localization stages."""
    
    def __init__(self, input_shape=(100, 100, 3), num_of_output = 4, steps_per_epoch = 50):
        self.input_shape = input_shape
        self.model = None
        self.num_of_output = num_of_output
        self.steps_per_epoch= steps_per_epoch
        # Keras convention: (height, width, channels) for channels_last
        self.image_height, self.image_width = self.input_shape[:2]
        self.alpha_bb = 1.0 # custom loss param: weight for bounding box coordinate loss
        self.beta_obj = 1.0 # custom loss param: weight for classification score loss
        self.gamma_obj = 0.5 # custom loss param: weight for objectness score loss

        # Grid dimensions — VGG16 with 5 max-pools shrinks spatial dims by 2^5 = 32
        # For 200×200 input: 200/32 = 6.25 → VGG outputs (6,6,512)
        self.grid_h = self.image_height // 32  # 6 for 200px
        self.grid_w = self.image_width // 32   # 6 for 200px
        self.num_cells = self.grid_h * self.grid_w  # 36 cells total

        # Anchor boxes: each cell makes num_anchors predictions instead of 1
        # Each anchor is (width, height) in image-relative coordinates [0,1]
        self.num_anchors = 3
        self.num_classes = 3
        self.anchors = [
            (0.20, 0.25),  # small  — covers ~1.0× scale objects
            (0.30, 0.40),  # medium — covers ~1.5× scale objects
            (0.45, 0.55),  # large  — covers ~2.0× scale objects
        ]
        self.total_anchors = self.num_cells * self.num_anchors  # 36 × 3 = 108

        # Compensate for sample-weight dilution:
        # With 108 anchor slots and only 1 matched slot per image, Keras averages
        # the sample-weighted loss over all 108 slots → bbox/class loss diluted by 108×.
        # Objectness has NO sample weight (all 108 slots train), so its gradient is
        # ~108× stronger. Scaling alpha/beta by total_anchors rebalances the gradients.
        self.alpha_bb *= self.total_anchors   # 1.0 × 108 = 108.0
        self.beta_obj *= self.total_anchors   # 1.0 × 108 = 108.0
    
    def build_vgg16_backbone_model(self, vgg_weights='imagenet', output_activation_func='sigmoid'):
        """Build common VGG16 backbone (no top, with custom head)."""
        tf = get_tf()
        VGG16 = get_vgg16()
        layers = get_keras_layers()
        
        vgg_base = VGG16(include_top=False, weights=vgg_weights, input_shape=self.input_shape)
        vgg_base.trainable = False

        #flatten the VGG output
        x = layers.Flatten()(vgg_base.output)

        # Hidden layers to learn complex mappings from VGG features to bbox + objectness
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        #create FC layer with the output
        #output is top-left corner (x1,y1) and height and width (h,w) --> (x1, y1, h, w, objectness)
        x = layers.Dense(self.num_of_output, activation = output_activation_func)(x)
      
        self.model = tf.keras.models.Model(vgg_base.input, x)

    def build_vgg16_backbone_multiclass_model(self, vgg_weights='imagenet', unfreeze_last_n_blocks=0):
        """Build common VGG16 backbone (no top, with custom head).

        Args:
            vgg_weights: Pretrained weights to use ('imagenet' or None).
            unfreeze_last_n_blocks: Number of VGG16 conv blocks to unfreeze from the top (0-5).
                0 = all frozen, 1 = block5, 2 = block4+block5, etc.
        """
        tf = get_tf()
        VGG16 = get_vgg16()
        layers = get_keras_layers()

        vgg_base = VGG16(include_top=False, weights=vgg_weights, input_shape=self.input_shape)
        vgg_base.trainable = False

        # Unfreeze the last N conv blocks for fine-tuning
        if unfreeze_last_n_blocks > 0:
            block_prefixes = ['block5', 'block4', 'block3', 'block2', 'block1']
            unfreeze_prefixes = block_prefixes[:unfreeze_last_n_blocks]
            for layer in vgg_base.layers:
                if any(layer.name.startswith(prefix) for prefix in unfreeze_prefixes):
                    layer.trainable = True

        # === Convolutional head: keeps (6,6) spatial grid alive ===
        # VGG output: (batch, 6, 6, 512) — no Flatten or GAP
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(vgg_base.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        # 1x1 Conv2D output heads with ANCHOR BOXES
        # Each cell predicts num_anchors boxes. Filters = num_anchors × values_per_anchor.
        # Reshape flattens (grid_h, grid_w, num_anchors*V) → (total_anchors, V)
        # where flat_idx = cell_idx * num_anchors + anchor_idx

        # Bbox: each anchor predicts [tx, ty, tw, th]
        bbox_conv = layers.Conv2D(self.num_anchors * 4, (1, 1), activation='sigmoid',
                                  name='bbox_conv')(x)
        # (6,6,12) → (108,4) — one [tx,ty,tw,th] per anchor slot
        bbox_output = layers.Reshape((self.total_anchors, 4), name='bbox_output')(bbox_conv)

        # Class: each anchor predicts a class distribution
        # NO activation on Conv2D — softmax must be applied AFTER reshape so each
        # group of num_classes values (per anchor) is normalized independently
        class_conv = layers.Conv2D(self.num_anchors * self.num_classes, (1, 1),
                                   name='class_conv')(x)
        # (6,6,9) → (108,3) then softmax on last dim → per-anchor class probabilities
        class_reshaped = layers.Reshape((self.total_anchors, self.num_classes))(class_conv)
        class_output = layers.Activation('softmax', name='class_output')(class_reshaped)

        # Objectness: each anchor predicts object existence
        obj_conv = layers.Conv2D(self.num_anchors, (1, 1), activation='sigmoid',
                                 name='obj_conv')(x)
        # (6,6,3) → (108,1) — one objectness score per anchor slot
        objectness_output = layers.Reshape((self.total_anchors, 1),
                                           name='objectness_output')(obj_conv)

        # Multi-output model with anchor boxes
        # Output shapes: bbox(batch,108,4), class(batch,108,3), obj(batch,108,1)
        self.model = tf.keras.models.Model(
            inputs=vgg_base.input,
            outputs=[bbox_output, class_output, objectness_output]
        )


    def custom_loss_for_non_objects(self):
        """Returns a custom loss function that uses instance-level alpha/beta weights.

        When object exists (y_true[:, -1] = 1): bbox loss is weighted by alpha_bb
        When no object (y_true[:, -1] = 0): bbox loss is zeroed out (only objectness matters)
        """
        alpha_bb = self.alpha_bb
        beta_obj = self.beta_obj

        def loss_fn(y_true, y_pred):
            bce = get_binary_crossentropy()
            bounding_box_loss = bce(y_true[:, :-1], y_pred[:, :-1])
            objectness_loss = bce(y_true[:, -1], y_pred[:, -1])
            # y_true[:, -1] is 1 when object exists, 0 when not → zeros out bbox loss for negatives
            total_loss = (alpha_bb * bounding_box_loss * y_true[:, -1]) + (beta_obj * objectness_loss)
            return total_loss

        loss_fn.__name__ = 'custom_loss_for_non_objects'
        return loss_fn
    
    def compile_model(self, loss_func='binary_crossentropy', lr=1e-3, metrics=None, loss_weights=None):
        """Compile the model with loss function and optimizer.

        Args:
            loss_func: Loss function name, or dict of {output_name: loss} for multi-output models.
            lr: Learning rate.
            metrics: Metrics to track.
            loss_weights: Optional dict of {output_name: weight} for multi-output models.
        """
        tf = get_tf()
        self.model.compile(
            loss=loss_func,
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=metrics,
            loss_weights=loss_weights
        )

    @abstractmethod
    def image_generator(batch_size=64):
        pass

    @abstractmethod
    def build_model(self):
        """Build complete model with backbone + stage-specific head."""
        pass
         
    @abstractmethod
    def train(self, dataset, epochs=1):
        """Stage-specific training logic."""
        pass