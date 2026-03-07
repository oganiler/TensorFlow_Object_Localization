# cli.py - main entrypoint
import sys
from typing import Optional
from .white_box_locator import WhiteBoxLocator
from .object_locator import ObjectLocator
from .imports import get_tf
from . import utils

def print_tf_version() -> None:
    tf = get_tf()
    print("TF Version", tf.__version__)

def execute_whitebox_detection():
    # Stage 1: white box localization
    print("\n=== Stage 1: White Box Localization ===")

    locator = WhiteBoxLocator(input_shape=(100, 100, 3), num_of_output = 4, steps_per_epoch = 50)

    print("\nBuild The Model")
    locator.build_model()

    print("\nCompile The Model")
    locator.compile_model(loss_func='binary_crossentropy', lr=1e-3, metrics=['accuracy'])

    print(locator.model.summary())
        
    print("\nFit The Model")
    history = locator.train(batch_size=64, epochs=3)

    print("\nPlot Training History")
    utils.plot_training_history(history)

    print("\nPredict and Visualize")
    locator.predict_and_visualize()

def execute_actual_object_detection():
    # Stage 2: actual object localization against black background
    print("\n=== Stage 2: Actual Object Localization ===")

    model_path = 'object_locator_model.keras'
    locator = ObjectLocator(input_shape=(200, 200, 3), num_of_output = 5, steps_per_epoch = 50,
                            actual_image_path = 'charmander-tight.png', backgrounds_dir='backgrounds')

    # Try to load a previously saved model; train only if none exists
    if not locator.load_model(model_path, custom_model=True):
        print("\nBuild The Model")
        locator.build_model()

        print("\nCompile The Model")
        locator.compile_model(loss_func=locator.custom_loss_for_non_objects(), lr=1e-4)

        print(locator.model.summary())

        print("\nFit The Model")
        history = locator.train(batch_size=64, epochs=30, model_path=model_path)

        print("\nPlot Training History")
        utils.plot_training_history(history)

    print("\nPredict and Visualize")
    locator.predict_and_visualize(batch_size=3)

def main(argv: Optional[list] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    try:
        print_tf_version()
        
        #execute_whitebox_detection()
        execute_actual_object_detection()
        
        return 0
    except Exception as ex:
        print("\nError in main:", ex, file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1