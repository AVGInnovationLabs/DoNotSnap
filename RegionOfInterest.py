import cv2
import math
import numpy as np
from util import pyramid, sliding_window


def boundingRects(scale, contours):
    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        yield [x * scale, y * scale, w * scale, h * scale]


def extractEdges(hue, intensity):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    edges = cv2.Canny(intensity, 120, 140)
    hue_edges = cv2.Canny(cv2.GaussianBlur(hue, (5, 5), 0), 0, 255)
    combined_edges = cv2.bitwise_or(hue_edges, edges)
    _, mask = cv2.threshold(combined_edges, 40, 255, cv2.THRESH_BINARY)
    return cv2.erode(cv2.GaussianBlur(mask, (3, 3), 0), kernel, iterations=1)


def roiFromEdges(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # close gaps in edges to create continous regions
    roi = cv2.dilate(edges, small_kernel, iterations=14)
    return cv2.erode(roi, kernel, iterations=4)


def findEllipses(edges):
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ellipseMask = np.zeros(edges.shape, dtype=np.uint8)
    contourMask = np.zeros(edges.shape, dtype=np.uint8)

    pi_4 = np.pi * 4

    for i, contour in enumerate(contours):
        if len(contour) < 5:
            continue

        area = cv2.contourArea(contour)
        if area <= 100:  # skip ellipses smaller then 10x10
            continue

        arclen = cv2.arcLength(contour, True)
        circularity = (pi_4 * area) / (arclen * arclen)
        ellipse = cv2.fitEllipse(contour)
        poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)

        # if contour is circular enough
        if circularity > 0.6:
            cv2.fillPoly(ellipseMask, [poly], 255)
            continue

        # if contour has enough similarity to an ellipse
        similarity = cv2.matchShapes(poly.reshape((poly.shape[0], 1, poly.shape[1])), contour, cv2.cv.CV_CONTOURS_MATCH_I2, 0)
        if similarity <= 0.2:
            cv2.fillPoly(contourMask, [poly], 255)

    return ellipseMask, contourMask


def findCircles(hue, intensity):
    houghCirclesMask = np.zeros(hue.shape, dtype=np.uint8)

    blurred_hue = cv2.GaussianBlur(hue, (9, 9), 2)
    blurred_intensity = cv2.GaussianBlur(intensity, (9, 9), 2)
    hue_circles = cv2.HoughCircles(blurred_hue, cv2.cv.CV_HOUGH_GRADIENT, 0.5, hue.shape[0] / 8, param1=10, param2=25, maxRadius=100)
    intensity_circles = cv2.HoughCircles(blurred_intensity, cv2.cv.CV_HOUGH_GRADIENT, 0.5, hue.shape[0] / 8, param1=185, param2=20, maxRadius=100)

    circles = np.vstack((hue_circles[0] if hue_circles is not None else np.empty((0, 3)),
                         intensity_circles[0] if intensity_circles is not None else np.empty((0, 3))))

    for (x, y, r) in circles:
        cv2.circle(houghCirclesMask, (int(round(x)), int(round(y))), int(round(r)), 255, -1)

    return houghCirclesMask


def findRANSACCircles(edges, circleSearches=5):
    edges = edges.copy()
    mask = np.zeros(edges.shape, dtype=np.uint8)

    minRadius = 10
    maxRadius = 100

    def verifyCircle(dt, center, radius):
        minInlierDist = 2.0
        maxInlierDistMax = 100.0
        maxInlierDist = max(minInlierDist, min(maxInlierDistMax, radius / 25.0))

        # choose samples along the circle and count inlier percentage
        samples = np.arange(0, 2 * np.pi, 0.05)
        cX = radius * np.cos(samples) + center[0]
        cY = radius * np.sin(samples) + center[1]

        coords = np.array((cX, cY)).T
        counter = len(samples)

        cXMask = (cX < dt.shape[1]) & (cX >= 0)
        cYMask = (cY < dt.shape[0]) & (cY >= 0)
        cMask = cXMask & cYMask

        gdt = dt[cY[cMask].astype(int), cX[cMask].astype(int)]
        dtMask = gdt < maxInlierDist

        inlierSet = coords[cMask][dtMask]
        inlier = len(inlierSet)

        return float(inlier) / counter, inlierSet

    def getCircle(p1, p2, p3):
        x1 = float(p1[0])
        x2 = float(p2[0])
        x3 = float(p3[0])

        y1 = float(p1[1])
        y2 = float(p2[1])
        y3 = float(p3[1])

        center_x = (x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)
        x = 2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2)
        if not x:
            return None, None
        center_x /= x

        center_y = (x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)
        y = 2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2)
        if not y:
            return None, None
        center_y /= y

        radius = math.sqrt((center_x - x1) * (center_x - x1) + (center_y - y1) * (center_y - y1))

        return (center_x, center_y), radius

    def getPointPositions(binaryImage):
        return [(x, y) for y, x in zip(*np.where(binaryImage > 0))]

    for _ in xrange(circleSearches):
        edgePositions = getPointPositions(edges)

        # create distance transform to efficiently evaluate distance to nearest edge
        dt = cv2.distanceTransform(255 - edges, cv2.cv.CV_DIST_L1, 3)

        bestCircleCenter = None
        bestCircleRadius = 0
        bestCirclePercentage = 0

        minCirclePercentage = 0.6  # at least 60% of a circle must be present

        maxNrOfIterations = len(edgePositions)  # TODO: adjust this parameter or include some real ransac criteria with inlier/outlier percentages to decide when to stop

        for its in xrange(maxNrOfIterations):
            # RANSAC: randomly choose 3 point and create a circle:

            # TODO: choose randomly but more intelligent,
            # so that it is more likely to choose three points of a circle.
            # For example if there are many small circles, it is unlikely to randomly choose 3 points of the same circle.
            idx1 = np.random.randint(len(edgePositions))
            idx2 = np.random.randint(len(edgePositions))
            idx3 = np.random.randint(len(edgePositions))

            # we need 3 different samples:
            if idx1 == idx2 or idx1 == idx3 or idx3 == idx2:
                continue

            # create circle from 3 points:
            center, radius = getCircle(edgePositions[idx1], edgePositions[idx2], edgePositions[idx3])
            if not center or radius > maxRadius:
                continue

            # inlier set unused at the moment but could be used to approximate a (more robust) circle from alle inlier
            # verify or falsify the circle by inlier counting:
            cPerc, inlierSet = verifyCircle(dt, center, radius)

            if cPerc >= bestCirclePercentage and radius >= minRadius:
                bestCirclePercentage = cPerc
                bestCircleRadius = radius
                bestCircleCenter = center

        # draw if good circle was found
        if bestCirclePercentage >= minCirclePercentage and bestCircleRadius >= minRadius:
            cv2.circle(mask, (int(round(bestCircleCenter[0])), int(round(bestCircleCenter[1]))), int(round(bestCircleRadius)), 255, -1)
            # mask found circle
            cv2.circle(edges, (int(round(bestCircleCenter[0])), int(round(bestCircleCenter[1]))), int(round(bestCircleRadius)), (0, 0, 0), 3)
    return mask


def weightMap(hue, intensity, edges, roi):
    ellipseMask, contourMask = findEllipses(edges)
    circlesMask = findCircles(hue, intensity)
    ransacMask = findRANSACCircles(edges)

    # create a map by combining different masks
    # circle/ellipse detection masks have a higher weight then roi mask and ransac mask
    combinedMask = np.zeros(edges.shape, dtype=np.float32)
    combinedMask += ellipseMask
    combinedMask += contourMask
    combinedMask += circlesMask
    combinedMask += ransacMask.astype(np.float32) / 2.0
    combinedMask += roi.astype(np.float32) / 4.0

    nonZeroMask = combinedMask != 0

    # rescale mask to 0 .. 1 range
    # (weight is either in 0.25 .. 1 range or 0)
    combinedMask[nonZeroMask] /= 255 * 3.75
    combinedMask[nonZeroMask] *= 0.6
    combinedMask[nonZeroMask] += 0.4
    return combinedMask


def roiMask(image, boundaries):
    scale = max([1.0, np.average(np.array(image.shape)[0:2] / 400.0)])
    shape = (int(round(image.shape[1] / scale)), int(round(image.shape[0] / scale)))

    small_color = cv2.resize(image, shape, interpolation=cv2.INTER_LINEAR)

    # reduce details and remove noise for better edge detection
    small_color = cv2.bilateralFilter(small_color, 8, 64, 64)
    small_color = cv2.pyrMeanShiftFiltering(small_color, 8, 64, maxLevel=1)
    small = cv2.cvtColor(small_color, cv2.COLOR_BGR2HSV)

    hue = small[::, ::, 0]
    intensity = cv2.cvtColor(small_color, cv2.COLOR_BGR2GRAY)

    edges = extractEdges(hue, intensity)
    roi = roiFromEdges(edges)
    weight_map = weightMap(hue, intensity, edges, roi)

    _, final_mask = cv2.threshold(roi, 5, 255, cv2.THRESH_BINARY)
    small = cv2.bitwise_and(small, small, mask=final_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

    for (lower, upper) in boundaries:
        lower = np.array([lower, 80, 50], dtype="uint8")
        upper = np.array([upper, 255, 255], dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(small, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        final_mask = cv2.bitwise_and(final_mask, mask)

    # blur the mask for better contour extraction
    final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
    return (final_mask, weight_map, scale)


def extractRoi(image, winSize, stepSize):
    # hue boundaries
    colors = [
        (15, 30)  # orange-yellow
    ]

    mask, weight_map, mask_scale = roiMask(image, colors)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yield weight_map, mask_scale

    for resized in pyramid(image, winSize):
        scale = image.shape[0] / resized.shape[0]
        for x, y, w, h in boundingRects(mask_scale, contours):
            x /= scale
            y /= scale
            w /= scale
            h /= scale
            center = (min(x + w / 2, resized.shape[1]), min(y + h / 2, resized.shape[0]))
            if w > winSize[0] or h > winSize[1]:
                for x, y, window in sliding_window(resized, (int(x), int(y), int(w), int(h)), stepSize, winSize):
                    yield ((x, y, winSize[0], winSize[1]), scale, window)
            else:
                x = max(0, int(center[0] - winSize[0] / 2))
                y = max(0, int(center[1] - winSize[1] / 2))
                window = resized[y:y + winSize[1], x:x + winSize[0]]
                yield ((x, y, winSize[0], winSize[1]), scale, window)
