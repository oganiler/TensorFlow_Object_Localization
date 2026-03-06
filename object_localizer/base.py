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
        self.alpha_bb = 2.0 # custom loss param: weight for bounding box coordinate loss
        self.beta_obj = 0.5 # custom loss param: weight for objectness score loss
    
    def build_vgg16_backbone_model(self, vgg_weights='imagenet', output_activation_func='sigmoid'):
        """Build common VGG16 backbone (no top, with custom head)."""
        tf = get_tf()
        VGG16 = get_vgg16()
        layers = get_keras_layers()
        
        vgg_base = VGG16(include_top=False, weights=vgg_weights, input_shape=self.input_shape)
        vgg_base.trainable = False

        #flatten the VGG output
        x = layers.Flatten()(vgg_base.output)

        #create FC layer with the output
        #output is top-left corner (x1,y1) and height and width (h,w) --> (x1, y1, h, w)
        x = layers.Dense(self.num_of_output, activation = output_activation_func)(x)
      
        self.model = tf.keras.models.Model(vgg_base.input, x)


    @staticmethod
    def custom_loss_for_non_objects(y_true, y_pred):
        """Custom loss to penalize false positives (non-object areas predicted as objects)."""
        bce = get_binary_crossentropy()
        
        # Calculate standard binary cross-entropy loss
        bounding_box_loss = bce(y_true[:, :-1], y_pred[:, :-1])  # Loss for bounding box coords
        objectness_loss = bce(y_true[:, -1], y_pred[:, -1])  # Loss for objectness score
        total_loss = (2.0 * bounding_box_loss * y_true[:, -1]) + (0.5 * objectness_loss)

        return total_loss

    def compile_model(self, loss_func='binary_crossentropy', lr=1e-3, metrics=None):
        """Compile the model with loss function and optimizer."""
        tf = get_tf()
        self.model.compile(
            loss=loss_func,
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=metrics
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