import re
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from deepface.commons.functions import initialize_detector, load_image, detect_face2
from deepface import DeepFace

detector_backend = 'mtcnn'
grayscale = False
enforce_detection = False
return_region = False
db_folder = "db"
col_dict = {"ZuoLin" : "Green", "JiaPeng": "Red"}

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
            new_im = Image.fromarray(detected_face)
            new_im.save("current.jpg")
            df = DeepFace.find(img_path = "current.jpg", db_path = "db", enforce_detection=False)
            df.sort_values('VGG-Face_cosine', inplace=True, ascending=True)
            face_detected = re.split(r' |/|\\', df['identity'].iloc[0])[1]
        except Exception as err:
            print("ERROR:", err)
            
        bboxs.append([x, y, w, h, detection['confidence'],face_detected])
        
        print(f"face detected: {face_detected}")

        face_count += 1
            
    output_img = drawBBoxs(img, bboxs, col_dict, col_dict.get(face_detected, "blue"))
    
    return output_img
    
    


def drawBBoxs(base_img, bboxs, col_dict, colour="blue"):
    output_img = base_img.copy()
    output_img = output_img[:, :, ::-1]

    output_img = Image.fromarray(np.uint8(output_img)).convert('RGB')
    draw = ImageDraw.Draw(output_img)

    for bbox in bboxs:
        try:
            colour = col_dict[bbox[5]]
        except:
            colour = "blue"
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