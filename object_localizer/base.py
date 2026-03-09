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

        # Global Average Pooling: reduces (h, w, 512) → (512,) with far fewer params than Flatten
        x = layers.GlobalAveragePooling2D()(vgg_base.output)

        # Shared hidden layers with BatchNorm for stable training
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        #seperate output heads for bounding box regression and multi-class classification and objecness
        bbox_output = layers.Dense(4, activation='sigmoid', name='bbox_output')(x) # Location
        class_output = layers.Dense(3, activation='softmax', name='class_output')(x) # Object class (3 classes in this example)
        objectness_output = layers.Dense(1, activation='sigmoid', name='objectness_output')(x) # Objectness score (0 or 1)

        # Multi-output model: each head is a separate output with its own loss
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