import cv2
import sys
import pickle
import numpy as np

from AffineInvariantFeatures import AffineInvariant
from RegionOfInterest import extractRoi
from TemplateMatcher import TemplateMatch, Templates  # imported to allow deserialization
from util import non_max_suppression_fast

from PIL import Image
from matplotlib import pyplot as plt


def classify(model, features, coords, weight_map, mask_scale):
    rects = []

    if len(features):
        y_pred = model.predict_proba(features)
        success = 0

        for (_, result), ((x, y, w, h), scale) in zip(y_pred, coords):
            mask_window = weight_map[y / mask_scale:(y + h) / mask_scale, x / mask_scale:(x + w) / mask_scale]
            weight = mask_window.max()
            if result * weight > 0.5:
                success += 1
                rects.append([x * scale, y * scale, (x + w) * scale, (y + h) * scale])

        print 'windows identified as positive: %d/%d(%0.2f%%)' % (success, len(y_pred), 100.0 * success / len(y_pred))
    else:
        print 'windows identified as positive: 0/%d(0%%)' % len(features)

    return np.array(rects)


def main(image_file):
    image = Image.open(image_file)
    if image is None:
        print 'Could not load image "%s"' % sys.argv[1]
        return

    image = np.array(image.convert('RGB'), dtype=np.uint8)
    image = image[:, :, ::-1].copy()

    winSize = (200, 200)
    stepSize = 32

    roi = extractRoi(image, winSize, stepSize)
    weight_map, mask_scale = next(roi)

    samples = [(rect, scale, cv2.cvtColor(window, cv2.COLOR_BGR2GRAY))
               for rect, scale, window in roi]

    X_test = [window for rect, scale, window in samples]
    coords = [(rect, scale) for rect, scale, window in samples]

    extractor = cv2.FeatureDetector_create('SURF')
    detector = cv2.DescriptorExtractor_create('SURF')

    affine = AffineInvariant(extractor, detector)

    saved = pickle.load(open('classifier.pkl', 'rb'))

    feature_transform = saved['pipe']
    model = saved['model']

    print 'Extracting Affine transform invariant features'
    affine_invariant_features = affine.transform(X_test)
    print 'Matching features with template'
    features = feature_transform.transform(affine_invariant_features)

    rects = classify(model, features, coords, weight_map, mask_scale)
    for (left, top, right, bottom) in non_max_suppression_fast(rects, 0.4):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 0), 10)
        cv2.rectangle(image, (left, top), (right, bottom), (32, 32, 255), 5)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':
    image_file = sys.argv[1] if len(sys.argv) >= 2 else 'sample.jpg'
    main(image_file)
