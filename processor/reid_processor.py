import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from reid_part import run_market as reid_model
import glob

#load model
model_path = 'reid_part/reid_model_market.h5'
model = reid_model.load_model(model_path)
root_dir = 'db_body/'
filepaths = [x.split('/') for x in glob.iglob(root_dir + '**/*.*', recursive=True)]
db = {}
for file in filepaths:
    if file[-2] not in db:
        db[file[-2]] = []
    db[file[-2]].append('/'.join(file))

def is_jpg(filename):
    try:
        i=Image.open(filename)
        return i.format =='JPEG'
    except IOError:
        return False
    
print(db)
for k,v in db.items():
    for vv in v:
#         with Image.open(vv) as im:
#             im.save(vv, "JPEG") 
        print(k, vv, is_jpg(vv))


def detect_body_cropped(cropped_img):
    """
    takes in a list of person bounding boxes and image
    crops the image to the bounding boxes and detects from database
    returns drawn img, person bounding boxes and id-ed name: [(x1, y1), (x2, y2), "ZuoLin"]
    """
    
    if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
        return "Unknown", 0
    
    new_im = Image.fromarray(cropped_img)
    new_im.save("current_body.jpg", "JPEG") 
    
    print("current_body.jpg", is_jpg("current_body.jpg"))

    best_identity = 'Unknown'
    best_average_confidence = 0.1 #can be some arbitrary v
    for identity in db:
        filenames = db[identity]
        pred, confidence = reid_model.check_images(model, "current_body.jpg", filenames)
        
        #different scoring functions here
#         average_confidence = np.mean(confidence[pred])
        # average_confidence = np.mean(confidence)
        temp = confidence[pred]
        if len(temp) > 0:
#             average_confidence = np.mean(confidence)
            average_confidence = max(confidence[pred])
        else:
            average_confidence = 0
        print('body:',identity, average_confidence)
        
        if average_confidence > best_average_confidence:
            best_average_confidence = average_confidence
            best_identity = identity
            
    return best_identity, best_average_confidence
    
            