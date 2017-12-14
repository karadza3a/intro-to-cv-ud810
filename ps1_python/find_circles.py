from ps1_python.hough_circles_acc import hough_circles_acc
from ps1_python.hough_peaks import hough_peaks


#     % Find circles in given radius range using Hough transform.
def find_circles(image, radii):
    return [(radius, hough_peaks(hough_circles_acc(image, radius), 10, 2*radius, verbose=False)) for radius in radii]
