# cli.py - main entrypoint
import sys
from typing import Optional
from .white_box_locator import WhiteBoxLocator
from .imports import get_tf

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
    locator.compile_model(loss_func='binary_crossentropy', lr=1e-3)

    print(locator.model.summary())
        
    print("\nFit The Model")
    locator.train(batch_size=64, epochs=2)

def main(argv: Optional[list] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    try:
        print_tf_version()
        
        execute_whitebox_detection()
        
        return 0
    except Exception as ex:
        print("\nError in main:", ex, file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1