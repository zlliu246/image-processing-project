import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import tensorflow as tf
from deepface.commons.functions import initialize_detector, load_image, detect_face2
import os

from processor import process
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="dataset/2fps")
parser.add_argument("-o", "--output", type=str, default="output")
parser.add_argument("-f", "--fps", type=int, default=2)


args = parser.parse_args()

# constants
INPUT_FOLDER = args.input
OUTPUT_FOLDER = args.output
FPS = args.fps

print(f"INPUT_FOLDER: {INPUT_FOLDER}")
print(f"OUTPUT_FOLDER: {OUTPUT_FOLDER}")
print(f"FPS: {FPS}")
print()

videos = os.listdir(INPUT_FOLDER)
for video in videos:
    print(f"PROCESSING VIDEO {video}")
    VIDEO_PATH = INPUT_FOLDER + "/" + video
    OUTPUT_PATH = OUTPUT_FOLDER + "/" + video

    current_frame = 1
    cap = cv2.VideoCapture(VIDEO_PATH)

    ret,frame = cap.read()
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc("M", "J", "P", "G"), FPS, (frame_width, frame_height))
    out.write(frame)

    # parsing video frame by frame
    while cap.isOpened():
        current_frame += 1
        print(f"PROCESSING FRAME {current_frame}")
        
        ret, frame = cap.read()
        if ret:
            
            # IMAGE FRAME GOES HERE
            img = load_image(frame)

            """
            ATTENTION: All your model processing goes into the "process" function
            """
            new_img = process(img)

            out.write(new_img)

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\ndone\n")