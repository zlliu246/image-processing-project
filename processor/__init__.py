from processor.deepface_processor import *
from processor.yolo_processor import detect_objects_yolo
from proximity.proximity_model import ProximityModel
from mtcnn import MTCNN

proximity_model = ProximityModel()

def process(img):
    """
    processes 1 image at a time, returns new image
    """

    # bboxs = [[20, 20, 20, 20, 0.99, "Laptop"]]
    # new_img = drawBBoxs(img, bboxs, col_dict)
    new_img = img
    person_face_detected = None

    try:
        new_img, person_boxes, object_boxes = detect_objects_yolo(new_img)
    except Exception as e:
        print(f"yolo ERROR, \n{e}")

    try:
        # Modified to detect on cropped images and output [person bbox, face_name] to track corresponding face to body
        new_img, person_face_detected = detect_deepface_cropped(new_img, person_boxes)
    except Exception as e:
        print(f"deepface ERROR, \n{e}")

    if person_face_detected and len(object_boxes) > 0:

        print("Running proximity model")
        # Only take first object bbox, assume only 1 object in frame
        proximity_model.update_curr(person_face_detected, object_boxes, img)
        proximity_model.detect_theft()

        print("person_face_detected:", person_face_detected)
        print("owners:", proximity_model.owners)
    new_img = proximity_model.draw_labels(new_img)

    return new_img
