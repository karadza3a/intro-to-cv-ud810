import numpy as np
import math
import matplotlib.pyplot as plt

#     % Draw lines found in an image using Hough transform.
#     %
#     % img: Image on top of which to draw lines
#     % outfile: Output image filename to save plot as
#     % peaks: Qx2 matrix containing row, column indices of the Q peaks found in accumulator

def line(xs, theta, rho):  # theta in radians
    if math.sin(theta) == 0:
        ys = xs
        return (rho - ys * math.sin(theta)) / math.cos(theta), ys
    return xs, (rho - xs * math.cos(theta)) / math.sin(theta)

def hough_lines_draw(im, peaks):
    for rho, theta in peaks:
        xs = np.array([0, im.shape[0]])
        xs, ys = line(xs, np.deg2rad(theta), rho)
        plt.plot(list(xs), ys, c="#00ff00")
