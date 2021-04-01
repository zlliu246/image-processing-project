import numpy as np
import cv2


def preprocess(img):
    image = np.clip(img/255, 0, 1)
    image = cv2.resize(image, (640, 480))
    image = image[None, :]
    return image


def postprocess(img, w, h):
    img = to_multichannel(img[0])
    return cv2.resize(np.uint8(img*255), (w, h), interpolation=cv2.INTER_AREA)


def depth_norm(x, max_depth):
    return max_depth / x


def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale
    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(depth_norm(predictions, max_depth=maxDepth), minDepth, maxDepth) / maxDepth


def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:, :, 0]
    return np.stack((i, i, i), axis=2)


def get_distance(bbox1: list, bbox2: list):
    (x1, y1), (x1b, y1b) = bbox1
    (x2, y2), (x2b, y2b) = bbox2
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:             # rectangles intersect
        return 0.


def draw_boxes(image, p1, p2, label_str, label, offset):
    image = np.float32(image)
    cv2.putText(image,
                label_str + ' ' + label,
                (p1[0], p1[1] - offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1e-3 * image.shape[0],
                (0, 0, 255), 2)

    return np.uint8(image)


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = {"x1": bb1[0][0],
           "y1": bb1[0][1],
           "x2": bb1[1][0],
           "y2": bb1[1][1]}
    bb2 = {"x1": bb2[0][0],
           "y1": bb2[0][1],
           "x2": bb2[1][0],
           "y2": bb2[1][1]}

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou