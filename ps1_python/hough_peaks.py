import numpy as np


#     % Find peaks in a Hough accumulator array.
#     % threshold: Threshold at which values of H are considered to be peaks
#     % n, m: Size of the suppression neighborhood
#     % http://www.mathworks.com/help/images/ref/houghpeaks.html
def hough_peaks(hough_acc, max_num_peaks, threshold=None, nhood=None):
    h = hough_acc.copy()
    if threshold is None:
        threshold = 0.5 * np.max(h)

    if nhood is None:
        n, m = h.shape[0] / 50., h.shape[1] / 50.
    else:
        n, m = nhood

    peaks = []
    for i in range(max_num_peaks):
        rho, theta = np.unravel_index(np.argmax(h), h.shape)
        if h[rho, theta] < threshold:
            break

        rh_l, rh_r = int(max(0, rho - n // 2)), int(min(rho + n // 2 + 1, int(h.shape[0])))
        th_l, th_r = int(theta - m / 2), int(theta + m / 2 + 1)
        for r in range(rh_l, rh_r):
            np.put(h[r, :], range(th_l, th_r), [0], mode='wrap')  # circular indexing
        peaks.append((rho, theta))

    return np.array(peaks)
