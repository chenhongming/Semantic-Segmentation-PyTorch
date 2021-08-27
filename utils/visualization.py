import cv2 as cv
import numpy as np


def vis(pred, image, color_map):
    pred = pred.astype('int')
    assert pred.shape[0] == image.shape[0], "pred.shape[0]: {} image.shape[0]: {}" .format(pred.shape[0], image.shape[0])
    assert pred.shape[1] == image.shape[1], "pred.shape[1]: {} image.shape[1]: {}" .format(pred.shape[1], image.shape[1])
    h, w = pred.shape
    value = np.unique(pred)
    out = np.zeros((h, w, 3))
    for i in value:
        out[np.where(pred == i)] = color_map[i]
    out = np.concatenate((image, out), 1).astype('uint8')
    out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
    return out
