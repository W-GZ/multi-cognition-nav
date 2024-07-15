# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
utils/initialization
"""


def notebook_init():
    # For  notebooks
    print('Checking setup...')
    from IPython import display  # to display images and clear console output

    from yolov3.utils.general import emojis
    from yolov3.utils.torch_utils import select_device  # imports

    display.clear_output()
    select_device(newline=False)
    print(emojis('Setup complete ✅'))
    return display
