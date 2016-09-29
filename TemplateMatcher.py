import cv2
import numpy as np
from multiprocessing.pool import ThreadPool
from sklearn.base import BaseEstimator, TransformerMixin

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing


class Templates():
    def __init__(self, features):
        self.features = features


class TemplateMatch(TransformerMixin, BaseEstimator):
    def __init__(self, templates, ratio=0.75):
        self.templates = templates
        self.ratio = ratio

        flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.pool = ThreadPool(processes=cv2.getNumberOfCPUs())

    def __getstate__(self):
        return dict(templates=self.templates, ratio=self.ratio)

    def __setstate__(self, state):
        self.templates = state['templates']
        self.ratio = state['ratio']

        flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})
        self.pool = ThreadPool(processes=1)  # cv2.getNumberOfCPUs())

    def feature(self, match):
        return match.distance

    def process(self, features):
        count = len(features)
        if not count:
            return [0, 0, 0, 0, 0, 0]

        split = [0.08, 0.12096, 0.16192, 0.20288, 0.24384, 0.2848, 1]
        return np.histogram(features, bins=split)[0]

    def match(self, sample):
        feature = []
        for template in self.templates.features:
            # we have to have at least 2 descriptors for 2 nearest neighbour search
            if len(sample) < 2:
                feature.extend([0, 0, 0, 0, 0, 0])
            else:
                if template.dtype != sample.dtype or template.shape[1] != sample.shape[1]:
                    print '!!!'
                    print template.shape, template.dtype, sample.shape, sample.dtype
                raw_matches = self.matcher.knnMatch(template, trainDescriptors=sample, k=2)
                matches = [self.feature(m[0])
                           for m in raw_matches
                           if len(m) == 2 and m[0].distance < m[1].distance * self.ratio]
                feature.extend(self.process(matches))
        return np.array(feature)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        l = dict(n=0)

        def f(image):
            l['n'] += 1
            if (100 * l['n'] / len(X)) % 5 == 0:
                print '\r%d%% - %d/%d' % (100 * l['n'] / len(X), l['n'], len(X)),
            return self.match(image)

        ires = self.pool.imap(f, X)
        return np.array(list(ires))
