import re
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from deepface.commons.functions import initialize_detector, load_image, detect_face2
from deepface import DeepFace
import processor.reid_processor as reid
import os
import time
import queue
from proximity.utils.utils import get_iou

detector_backend = 'mtcnn'
grayscale = False
enforce_detection = False
return_region = False
db_folder = "db_face"
col_dict = {"ZuoLin" : "Green", "JiaPeng": "Red", "Cheng": "Blue"}


class Ballot:
    def __init__(self, name, coords, o_coords):
        self.name = name
        self.arr = []
        self.coords = coords
        self.o_coords = o_coords
        self.info = None
        self.best_count = 0
        
    def _count_votes(self):
        counter = {}
        best_vote = None
        best_count = 0
        for vote in self.arr:
            if vote not in counter:
                counter[vote] = 0
            counter[vote] += 1
            if counter[vote] > best_count:
                best_count = counter[vote]
                best_vote = vote
        print('voting:', counter)
        return best_vote, best_count
                
    def vote(self, face_detected):
        if face_detected == None:
            return self.name, self.best_count
        if len(self.arr) >= 20:
            del self.arr[0]
        self.arr.append(face_detected)
        self.name, self.best_count = self._count_votes()
        return self.name, self.best_count
    
    def iou(self, bb):
        return get_iou(self.o_coords,bb)
    def set_coords(self, bb):
        self.coords = bb
    def set_o_coords(self, bb):
        self.o_coords = bb
    def set_info(self, info):
    #[xmin+x, ymin+y, w, h, detections[0]['confidence'], face_detected]
        self.info = info
        self.info[-1] = self.name
        self.info[-2] = self.best_count/20
    
def assign_new_to_old(person_face_detected, old_ballots, info, o_coords):
    """
    takes in a list of person bounding boxes and identity [(xmin, ymin), (xmax, ymax), face_detected]
    try to match new bounding box to old
    if more new than old: create new ballot
    if less new than old: discard ballot
    """
    new_ballots = []
    votes = []
    completed = set()
    
    index = 0
    for i in range(len(person_face_detected)):
        bb = [person_face_detected[i][0], person_face_detected[i][1]]
        idt = person_face_detected[i][-1]
        best = None
        best_iou = 0.25
        for old in old_ballots:
            if old.name in completed:
                continue
            iou = old.iou(o_coords[i])
            if iou > best_iou:
                best = old
                best_iou = iou
        if best == None and idt != None:
            best = Ballot(idt, bb, o_coords[i]) if idt not in completed else None
        if best != None:
            if info[i][0] != None:
                best.vote(idt)
            best.vote(idt)
            completed.add(best.name)
            best.set_info(info[i])
            best.set_coords(bb)
            best.set_o_coords(o_coords[i])
            new_ballots.append(best)
    
    return new_ballots
    
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


def detect_deepface_cropped(img, person_boxes, old_ballots, original_boxes):
    """
    takes in a list of person bounding boxes and image
    crops the image to the bounding boxes and detects from database
    returns drawn img, person bounding boxes and id-ed name: [(x1, y1), (x2, y2), "ZuoLin"]
    """
    bboxs = []
    person_face_detected = []
    output_img = img.copy()
    for i in range(len(person_boxes)):
        p_bbox = person_boxes[i]
        o_bbox = original_boxes[i]
        (xmin, ymin), (xmax, ymax) = p_bbox
        cropped_img = img[ymin:ymax, xmin:xmax]
        detections = detect_face2(img=cropped_img, detector_backend=detector_backend, grayscale=grayscale, enforce_detection=enforce_detection)
        (oxmin, oymin), (oxmax, oymax) = p_bbox
        o_cropped_img = img[oymin:oymax, oxmin:oxmax]
        
        face_detected = None
        if len(detections) == 1:  # Only retain images that has 1 detected face
            
            x, y, w, h = detections[0]["box"]
            y_buffer = int(h * 0.5)
            x_buffer = int(w * 0.5)
            detected_face = cropped_img[max(0,int(y-y_buffer)):min(cropped_img.shape[0],int(y+h+y_buffer)), max(0,int(x-x_buffer)):min(cropped_img.shape[1],int(x+w+x_buffer))]

            try:
                ## calls reid_processor to confirm identity
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
                if dist > 0.035:
                    face_detected = None
                    pass
                else:
                    print('p detected by face:', face_detected)

            except Exception as err:
                print("ERROR:", err)
                
        if face_detected == None:
            best_identity, best_confidence = reid.detect_body_cropped(o_cropped_img[:, :, ::-1])
            face_detected = best_identity
            print('p detected by body:', face_detected)
            bboxs.append([None, None, None, None, best_confidence, face_detected])
        else:
            bboxs.append([xmin+x, ymin+y, w, h, detections[0]['confidence'], face_detected])
        person_face_detected.append([(xmin, ymin), (xmax, ymax), face_detected])
    
    old_ballots = assign_new_to_old(person_face_detected, old_ballots, bboxs, original_boxes)
    for face_detected in old_ballots:
        print(f"person detected: {face_detected.name}", f"confidence: {face_detected.best_count}")
        if face_detected.info[0] is not None:
            output_img = drawBBox(output_img, face_detected.info, col_dict, "yellow")
        cc = face_detected.o_coords
        output_img = drawBBox2(output_img, cc[0][0], cc[1][0], cc[0][1], cc[1][1], face_detected.name, col_dict, "yellow")
    
    person_detected = [[x.coords[0],x.coords[1],x.name] for x in old_ballots if x.name]
    return output_img, person_detected, old_ballots


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

def drawBBox(base_img, bbox, col_dict, colour="yellow"):
    #for only 1 bbox
    output_img = base_img.copy()
    output_img = output_img[:, :, ::-1]

    output_img = Image.fromarray(np.uint8(output_img)).convert('RGB')
    draw = ImageDraw.Draw(output_img)

    
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

def drawBBox2(base_img, x0,x1,y0,y1, name, col_dict, colour="yellow"):
    #for only 1 bbox
    output_img = base_img.copy()
    output_img = output_img[:, :, ::-1]

    output_img = Image.fromarray(np.uint8(output_img)).convert('RGB')
    draw = ImageDraw.Draw(output_img)

    
    try:
        colour = col_dict[name]
    except:
        colour = "yellow"
   
    draw.rectangle((x0, y0, x1, y1), outline=colour, width=5)
    draw.text((x0 + 10, y0 + 10), name, fill=colour)
#         draw.text((x0 + 10, y0 + 10), str(round(bbox[4],2)), fill=colour)

    output_img = np.array(output_img)
    output_img = output_img[:, :, ::-1]
    return output_img