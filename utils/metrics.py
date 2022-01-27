import numpy as np


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(preds, label, num_class, ignore_index=-1):
    preds = np.asarray(preds).copy()
    label = np.asarray(label).copy()

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    preds = preds * (label != ignore_index)

    # Compute area intersection:
    intersection = preds * (preds == label)
    (area_intersection, _) = np.histogram(intersection, bins=num_class-1, range=(1, num_class))

    # Compute area union:
    (area_pred, _) = np.histogram(preds, bins=num_class-1, range=(1, num_class))
    (area_lab, _) = np.histogram(label, bins=num_class-1, range=(1, num_class))
    area_union = area_pred + area_lab - area_intersection

    return area_intersection, area_union
