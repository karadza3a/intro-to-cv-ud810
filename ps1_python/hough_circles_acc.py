import numpy as np


#     % Compute Hough accumulator array for finding circles.
#     %
#     % BW: Binary (black and white) image containing edge pixels
#     % radius: Radius of circles to look for, in pixels
# end

def hough_circles_acc(image, radius):
    acc = np.zeros(image.shape, "uint")
    n, m = image.shape
    radius2 = radius * radius

    ys, xs = np.nonzero(image)

    for (x, y) in zip(xs, ys):
        lower = max(0, x - radius)
        upper = min(m, x + radius + 1)
        c_xs = np.arange(lower, upper)
        rhs = np.sqrt(radius2 - (x - c_xs) ** 2)
        c_y1 = np.round(y - rhs).astype("int")
        c_y2 = np.round(y + rhs).astype("int")

        y1_mask = (0 <= c_y1) & (c_y1 < n)
        y2_mask = (0 <= c_y2) & (c_y2 < n)

        acc[c_y1[y1_mask], c_xs[y1_mask]] += 1
        acc[c_y2[y2_mask], c_xs[y2_mask]] += 1

    return acc
