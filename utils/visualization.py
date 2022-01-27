import cv2 as cv
import numpy as np


# visualization predict result to BGR image
def vis(pred, image, color_map, is_merge):
    pred = pred.astype('int')
    assert pred.shape[0] == image.shape[0], "pred.shape[0]: {} image.shape[0]: {}" .format(pred.shape[0], image.shape[0])
    assert pred.shape[1] == image.shape[1], "pred.shape[1]: {} image.shape[1]: {}" .format(pred.shape[1], image.shape[1])
    h, w = pred.shape
    value = np.unique(pred)
    out = np.zeros((h, w, 3))
    for i in value:
        out[np.where(pred == i)] = color_map[i]
    if is_merge:
        out = cv.addWeighted(image, 0.7, out.astype('uint8'), 0.3, 0)
    else:
        out = np.concatenate((image, out), 1).astype('uint8')
    out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
    return out
