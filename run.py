import warnings
warnings.filterwarnings("ignore")

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="input", type=str, metavar='<str>', help='Input Folder', default="./REDUCED_VIDEOS")
parser.add_argument("-v", "--video", dest="video", type=str, metavar='<str>', help='Video Path for single video')
parser.add_argument("-g", "--gpu", dest="gpu", type=str, metavar='<str>', help='Select GPU by PCI_BUS_ID', default="0")
parser.add_argument("-o", "--output", dest="output", type=str, metavar='<str>', help='Output Folder', default="output")
parser.add_argument("-f", "--fps", dest="fps", type=int, metavar='<int>', help='FPS for output video', default=10)

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import pandas as pd
import numpy as np
import re
import cv2
from PIL import Image, ImageDraw
import tensorflow as tf
from deepface.commons.functions import load_image, detect_face2

from processor import process

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


for video in videos:
    print(f"PROCESSING VIDEO {video}")
    VIDEO_PATH = INPUT_FOLDER + "/" + video
    OUTPUT_PATH = OUTPUT_FOLDER + "/" + video.replace(".mp4", ".avi")  # My windows machine couldnt view in mp4, can change accordingly

    current_frame = 1
    cap = cv2.VideoCapture(VIDEO_PATH)

    ret,frame = cap.read()
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc("M", "J", "P", "G"), FPS, (frame_width, frame_height))
    out.write(frame)
    
    old_ballots = []

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
            new_img, old_ballots = process(img, old_ballots)

            out.write(new_img)

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\ndone\n")