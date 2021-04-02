import re
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from deepface.commons.functions import initialize_detector, load_image, detect_face2
from deepface import DeepFace
# import processor.reid_processor as reid
import os
import time

detector_backend = 'mtcnn'
grayscale = False
enforce_detection = False
return_region = False
db_folder = "db_face"
col_dict = {"ZuoLin" : "Green", "JiaPeng": "Red", "Cheng": "Blue"}


def detect_deepface(img):
    detections = detect_face2(img=img, detector_backend=detector_backend, grayscale=grayscale, enforce_detection=enforce_detection)

    if len(detections) == 0:
        return img

    bboxs = []
    face_count = 0
    for detection in detections:
        x, y, w, h = detection["box"]
        detected_face = img[int(y):int(y+h), int(x):int(x+w)]
        face_detected = "Unknown"
        
        try:
            detected_face = detected_face[:, :, ::-1]
            new_im = Image.fromarray(detected_face)
            tmp_image = str(int(time.time())) + ".jpg"
            new_im.save(tmp_image)
            df = DeepFace.find(img_path = tmp_image, db_path = db_folder, enforce_detection=False)
            os.remove(tmp_image)
            face_detected = re.split(r' |/|\\', df['identity'].iloc[0])[1]
        except Exception as err:
            print("ERROR:", err)
            
        bboxs.append([x, y, w, h, detection['confidence'],face_detected])
        
        print(f"face detected: {face_detected}")

        face_count += 1
            
    output_img = drawBBoxs(img, bboxs, col_dict, col_dict.get(face_detected, "yellow"))
    
    return output_img, face_detected


def detect_deepface_cropped(img, person_boxes):
    """
    takes in a list of person bounding boxes and image
    crops the image to the bounding boxes and detects from database
    returns drawn img, person bounding boxes and id-ed name: [(x1, y1), (x2, y2), "ZuoLin"]
    """
    bboxs = []
    person_face_detected = []
    output_img = img.copy()
    for p_bbox in person_boxes:
        (xmin, ymin), (xmax, ymax) = p_bbox
        cropped_img = img[ymin:ymax, xmin:xmax]
        detections = detect_face2(img=cropped_img, detector_backend=detector_backend, grayscale=grayscale, enforce_detection=enforce_detection)

        if len(detections) != 1:  # Only retain images that has 1 detected face
            continue

        x, y, w, h = detections[0]["box"]
        detected_face = cropped_img[int(y):int(y+h), int(x):int(x+w)]
        face_detected = None

        try:
            ## calls reid_processor to confirm identity
            # best_body_guess, body_confidence = reid.detect_body_cropped(cropped_img)
            # best_body_guess, body_confidence = reid.detect_body_cropped(cropped_img)
            detected_face = detected_face[:, :, ::-1]
            new_im = Image.fromarray(detected_face)
            tmp_image = str(int(time.time())) + ".jpg"
            new_im.save(tmp_image)
            df = DeepFace.find(img_path = tmp_image, db_path = db_folder, enforce_detection=False)
            os.remove(tmp_image)
            df.sort_values('VGG-Face_cosine', inplace=True, ascending=True)
            face_detected = re.split(r' |/|\\', df['identity'].iloc[0])[1]
            dist = float(df['VGG-Face_cosine'].iloc[0])
            
            ## need code when face id is not confident, use reid to confirm/dispute
            if dist > 0.025:
                print("face id is not confident")

        except Exception as err:
            print("ERROR:", err)

        bboxs.append([xmin+x, ymin+y, w, h, detections[0]['confidence'], face_detected])
        person_face_detected.append([(xmin, ymin), (xmax, ymax), face_detected])

        print(f"face detected: {face_detected}")
        if face_detected is not None:
            output_img = drawBBoxs(output_img, bboxs, col_dict, col_dict.get(face_detected, "yellow"))
    return output_img, person_face_detected


def drawBBoxs(base_img, bboxs, col_dict, colour="yellow"):
    output_img = base_img.copy()
    output_img = output_img[:, :, ::-1]

    output_img = Image.fromarray(np.uint8(output_img)).convert('RGB')
    draw = ImageDraw.Draw(output_img)

    for bbox in bboxs:
        try:
            colour = col_dict[bbox[5]]
        except:
            colour = "yellow"
        x0 = bbox[0]
        x1 = bbox[0] + bbox[2]
        y0 = bbox[1]
        y1 = bbox[1] + bbox[3]
        draw.rectangle((x0, y0, x1, y1), outline=colour, width=5)
        draw.text((x0 + 10, y0 + 10), bbox[5], fill=colour)
#         draw.text((x0 + 10, y0 + 10), str(round(bbox[4],2)), fill=colour)

    output_img = np.array(output_img)
    output_img = output_img[:, :, ::-1]
    return output_img

def getIOU(x, y):
    x_x1 = x[0]
    x_x2 = x_x1 + x[2]
    x_y1 = x[1]
    x_y2 = x_y1 + x[3]

    y_x1 = y[0]
    y_x2 = y_x1 + y[2]
    y_y1 = y[1]
    y_y2 = y_y1 + y[3]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(x_x1, y_x1)
    yA = max(x_y1, y_y1)
    xB = min(x_x2, y_x2)
    yB = min(x_y2, y_y2)

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # 0 , 1 , 2 , 3
    # x1, y1, x2, y2
	  # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (x_x2 - x_x1 + 1) * (x_y2 - x_y1 + 1)
    boxBArea = (y_x2 - y_x1 + 1) * (y_y2 - y_y1 + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou