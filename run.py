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
parser.add_argument("-i", "--input", type=str, help='Input Folder', default="dataset/2fps")
parser.add_argument("-v", "--video", type=str, help='Video Path for single video', default="NA")
parser.add_argument("-g", "--gpu", type=str, help='Select GPU by PCI_BUS_ID', default="NA")
parser.add_argument("-o", "--output", type=str, help='Output Folder', default="output")
parser.add_argument("-f", "--fps", type=int, help='FPS for output video', default=2)


args = parser.parse_args()

# constants
INPUT_FOLDER = args.input
OUTPUT_FOLDER = args.output
FPS = args.fps

try:
    os.mkdir(OUTPUT_FOLDER)
    print("OUTPUT_FOLDER created")
except:
    print("OUTPUT_FOLDER existed")

print(f"INPUT_FOLDER: {INPUT_FOLDER}")
print(f"OUTPUT_FOLDER: {OUTPUT_FOLDER}")
print(f"FPS: {FPS}")
print()

if args.video == "NA":
    videos = os.listdir(INPUT_FOLDER)
else:
    videos = [args.video]

if args.gpu != "NA":
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


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