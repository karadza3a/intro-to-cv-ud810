import math
import numpy as np


# black and white image containing edge pixels
def hough_lines_acc(image):
    n, m = image.shape[0] - 1, image.shape[1] - 1
    max_rho = int(math.ceil(math.sqrt(n * n + m * m)))
    acc = np.zeros((max_rho * 2, 180), "uint8")

    thetas = np.deg2rad(np.arange(0, 180))
    theta_sins = np.sin(thetas)
    theta_coss = np.cos(thetas)

    xs, ys = np.nonzero(image)

    for (x, y) in zip(xs, ys):
        rho_candidates = np.round((x+1) * theta_coss + (y+1) * theta_sins).astype("uint8")
        for (theta,), rho in np.ndenumerate(rho_candidates):
            if -max_rho <= rho < max_rho:
                acc[rho + max_rho][theta] += 1
    acc = acc - np.min(acc)
    return np.round(acc * (255 / np.max(acc))).astype("uint8"), max_rho
