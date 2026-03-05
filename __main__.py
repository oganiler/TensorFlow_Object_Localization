# __main__.py
# your existing code below
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #ignores some warnings

from object_localizer.cli import main

if __name__ == "__main__":
    main()