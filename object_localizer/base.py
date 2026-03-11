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

        self.num_classes = 3

        # === SSD-style multi-scale detection ===
        # Each VGG16 pooling layer produces a feature map at a different resolution.
        # Smaller strides → finer grids → better for detecting small objects.
        # Each scale gets 1 anchor tuned to the object sizes it's best at detecting.
        #
        # VGG16 pool layers for 200×200 input:
        #   block3_pool: 25×25 (stride  8) — small objects
        #   block4_pool: 12×12 (stride 16) — medium objects
        #   block5_pool:  6×6  (stride 32) — large objects
        self.scales = [
            {'name': 'block3_pool',
             'grid_h': self.image_height // 8,   # 25
             'grid_w': self.image_width  // 8,   # 25
             'anchors': [(0.15, 0.20)]},          # small objects
            {'name': 'block4_pool',
             'grid_h': self.image_height // 16,  # 12
             'grid_w': self.image_width  // 16,  # 12
             'anchors': [(0.30, 0.40)]},          # medium objects
            {'name': 'block5_pool',
             'grid_h': self.image_height // 32,  # 6
             'grid_w': self.image_width  // 32,  # 6
             'anchors': [(0.50, 0.55)]},          # large objects
        ]

        # Precompute per-scale metadata and cumulative flat-index offsets
        # Flat layout: [block3 slots | block4 slots | block5 slots]
        #              [0..624       | 625..768     | 769..804    ]
        offset = 0
        for s in self.scales:
            s['num_cells']   = s['grid_h'] * s['grid_w']
            s['num_anchors'] = len(s['anchors'])
            s['num_slots']   = s['num_cells'] * s['num_anchors']
            s['offset']      = offset
            offset += s['num_slots']
        self.total_anchors = offset  # 625 + 144 + 36 = 805

        # Objectness imbalance fix — auto-scale to maintain ~1:10 pos/neg ratio.
        # With 805 slots and 1 positive per image, hardcoded 0.1 would give
        # 804×0.1 = 80.4 (way too much negative weight). Auto-scaling fixes this:
        self.noobj_weight = 10.0 / (self.total_anchors - 1)
        # 805 slots → 10.0/804 ≈ 0.012   (effective ratio 1:10)
        # 108 slots → 10.0/107 ≈ 0.093   (same formula works for old config)

        # Compensate for sample-weight dilution:
        # With N anchor slots and only 1 matched slot per image, Keras averages
        # the sample-weighted loss over all N slots → bbox/class loss diluted by N×.
        self.alpha_bb *= self.total_anchors   # 1.0 × 805 = 805.0
        self.beta_obj *= self.total_anchors   # 1.0 × 805 = 805.0
    
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
        """Build SSD-style multi-scale VGG16 model.

        Taps into block3_pool, block4_pool, and block5_pool feature maps.
        Each scale gets its own independent conv head → bbox/class/objectness predictions.
        All scales are concatenated into unified output tensors.

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

        # === SSD-style: extract intermediate feature maps from VGG16 ===
        # Each pool layer provides features at a different spatial resolution:
        #   block3_pool → (25, 25, 256)  stride  8 — fine grid for small objects
        #   block4_pool → (12, 12, 512)  stride 16 — medium grid
        #   block5_pool → ( 6,  6, 512)  stride 32 — coarse grid for large objects
        feature_maps = [vgg_base.get_layer(s['name']).output for s in self.scales]

        # Build per-scale prediction heads, then concatenate across scales
        all_bbox, all_class, all_obj = [], [], []

        for scale, feat in zip(self.scales, feature_maps):
            sn = scale['name']  # e.g. 'block3_pool' — used as name prefix
            na = scale['num_anchors']  # 1 per scale
            ns = scale['num_slots']    # grid_h × grid_w × num_anchors

            # Per-scale convolutional head (independent weights per scale)
            h = layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                              name=f'{sn}_conv1')(feat)
            h = layers.BatchNormalization(name=f'{sn}_bn1')(h)
            h = layers.Dropout(0.3, name=f'{sn}_drop1')(h)
            h = layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                              name=f'{sn}_conv2')(h)
            h = layers.BatchNormalization(name=f'{sn}_bn2')(h)
            h = layers.Dropout(0.3, name=f'{sn}_drop2')(h)

            # Bbox head: [tx, ty, tw, th] per anchor, sigmoid to [0,1]
            bbox = layers.Conv2D(na * 4, (1, 1), activation='sigmoid',
                                 name=f'{sn}_bbox_conv')(h)
            bbox = layers.Reshape((ns, 4), name=f'{sn}_bbox')(bbox)
            all_bbox.append(bbox)

            # Class head: no activation on Conv2D → reshape → softmax per anchor
            cls = layers.Conv2D(na * self.num_classes, (1, 1),
                                name=f'{sn}_class_conv')(h)
            cls = layers.Reshape((ns, self.num_classes),
                                 name=f'{sn}_class_reshape')(cls)
            cls = layers.Activation('softmax', name=f'{sn}_class')(cls)
            all_class.append(cls)

            # Objectness head: sigmoid, 1 score per anchor
            obj = layers.Conv2D(na, (1, 1), activation='sigmoid',
                                name=f'{sn}_obj_conv')(h)
            obj = layers.Reshape((ns, 1), name=f'{sn}_obj')(obj)
            all_obj.append(obj)

        # Concatenate all scales → unified output tensors
        # Flat layout: [block3 slots | block4 slots | block5 slots]
        # Output shapes: bbox(batch,805,4), class(batch,805,3), obj(batch,805,1)
        bbox_output = layers.Concatenate(axis=1, name='bbox_output')(all_bbox)
        class_output = layers.Concatenate(axis=1, name='class_output')(all_class)
        objectness_output = layers.Concatenate(axis=1,
                                               name='objectness_output')(all_obj)

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