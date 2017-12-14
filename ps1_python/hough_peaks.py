import numpy as np


#     % Find peaks in a Hough accumulator array.
#     % threshold: Threshold at which values of H are considered to be peaks
#     % n, m: Size of the suppression neighborhood
#     % http://www.mathworks.com/help/images/ref/houghpeaks.html
def hough_peaks(hough_acc, max_num_peaks, threshold=None, nhood=None, verbose=False):
    h = hough_acc.copy()
    if threshold is None:
        threshold = 0.5 * np.max(h)

    if nhood is None:
        n, m = h.shape[0] / 50., h.shape[1] / 50.
    else:
        n, m = nhood
    n2, m2 = int(n // 2), int(m // 2)

    peaks = []
    for i in range(max_num_peaks):
        rho, theta = np.unravel_index(np.argmax(h), h.shape)
        if verbose:
            print("%5d %5d %5.3f" % (rho, theta, h[rho, theta]))
        if h[rho, theta] < threshold:
            break

        rh_l, rh_r = max(0, rho - n2), min(rho + n2 + 1, h.shape[0])
        th_l, th_r = max(0, theta - m2), min(theta + m2 + 1, h.shape[1])
        h[rh_l:rh_r, th_l:th_r] = 0
        peaks.append((rho, theta))

    return np.array(peaks)
