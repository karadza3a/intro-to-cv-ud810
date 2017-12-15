import math
import numpy as np


# black and white image containing edge pixels
def hough_lines_acc(image):
    n, m = image.shape[0], image.shape[1]
    max_rho = int(math.ceil(math.sqrt(n * n + m * m)))
    acc = np.zeros((max_rho * 2, 180), "uint")

    thetas_deg = np.arange(0, 180)
    thetas = np.deg2rad(thetas_deg)
    theta_sins = np.sin(thetas)
    theta_coss = np.cos(thetas)

    ys, xs = np.nonzero(image)

    for (x, y) in zip(xs, ys):
        rho_candidates = np.round(x * theta_coss + y * theta_sins).astype("int")
        mask = (-max_rho <= rho_candidates) & (rho_candidates < max_rho)
        acc[rho_candidates[mask] + max_rho, thetas_deg[mask]] += 1

    return acc, max_rho
