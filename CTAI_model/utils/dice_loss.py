import numpy as np


def dice(im1, im2):

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # 俩都为全黑
    if not (im1.any() or im2.any()):
        return 1.0

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    res = 2. * intersection.sum() / (im1.sum() + im2.sum())
    return np.round(res, 5)
