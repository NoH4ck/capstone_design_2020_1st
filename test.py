from darkflow.cli import cliHandler
import darkflow.net.yolov2.predict

def detect():
    if darkflow.net.yolov2.predict.detection_choice == 1:
        arg = [
            'flow', '--imgdir', './static/',
            '--model', './cfg/fingertip-yolo.cfg', '--load', '-1', '--batch', '1',
            '--threshold', '0.5'
        ]
    elif darkflow.net.yolov2.predict.detection_choice == 2:
        arg = [
            'flow', '--imgdir', './static/',
            '--model', './cfg/pupils_yolo.cfg', '--load', '-1', '--batch', '1',
            '--threshold', '0.9'
        ]

    cliHandler(arg)
