import cv2
import numpy as np
import itertools as it
from multiprocessing.pool import ThreadPool
from sklearn.base import BaseEstimator, TransformerMixin


class AffineInvariant(TransformerMixin, BaseEstimator):
    def __init__(self, detector, extractor):
        self.detector = detector
        self.extractor = extractor
        self.pool = ThreadPool(processes=cv2.getNumberOfCPUs())

    def affine_skew(self, tilt, phi, img, mask=None):
        h, w = img.shape[:2]
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
            mask[:] = 255
        A = np.float32([[1, 0, 0], [0, 1, 0]])
        if phi != 0.0:
            phi = np.deg2rad(phi)
            s, c = np.sin(phi), np.cos(phi)
            A = np.float32([[c, -s], [s, c]])
            corners = [[0, 0], [w, 0], [w, h], [0, h]]
            tcorners = np.int32(np.dot(corners, A.T))
            x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
            A = np.hstack([A, [[-x], [-y]]])
            img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if tilt != 1.0:
            s = 0.8*np.sqrt(tilt * tilt - 1)
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
            img = cv2.resize(img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
            A[0] /= tilt
        if phi != 0.0 or tilt != 1.0:
            h, w = img.shape[:2]
            mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
        Ai = cv2.invertAffineTransform(A)
        return img, mask, Ai

    def affine_detect(self, img, mask=None):
        params = [(1.0, 0.0)]
        for t in 2 ** (0.5 * np.arange(1, 6)):
            for phi in np.arange(0, 180, 72.0 / t):
                params.append((t, phi))

        def f(p):
            t, phi = p
            timg, tmask, Ai = self.affine_skew(t, phi, img)
            keypoints = self.detector.detect(timg, tmask)
            keypoints, descrs = self.extractor.compute(timg, keypoints)
            for kp in keypoints:
                x, y = kp.pt
                kp.pt = tuple(np.dot(Ai, (x, y, 1)))
            if descrs is None:
                descrs = np.zeros((0, 64), 'float32')
            return keypoints, descrs

        keypoints, descrs = [], []
        ires = it.imap(f, params)

        for i, (k, d) in enumerate(ires):
            keypoints.extend(k)
            descrs.extend(d)

        return keypoints, np.array(descrs)

    def extract_features(self, image):
        keypoints, descriptors = self.affine_detect(image)
        if descriptors is None or not len(descriptors):
            # treat failure to classify as negative sample
            return np.zeros((0, 64), 'float32')

        return descriptors

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        l = dict(n=0)

        def f(image):
            l['n'] += 1
            if (100 * l['n'] / len(X)) % 5 == 0:
                print '\r%d%% - %d/%d' % (100 * l['n'] / len(X), l['n'], len(X)),
            return self.extract_features(image)

        ires = self.pool.imap(f, X)
        return np.array(list(ires))
