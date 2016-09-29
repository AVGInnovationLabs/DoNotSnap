import cv2
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from AffineInvariantFeatures import AffineInvariant
from TemplateMatcher import TemplateMatch, Templates

from PIL import Image
from itertools import izip_longest

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz, DecisionTreeClassifier


def line_count(filename):
    with open(filename) as data:
        return sum(1 for line in data)


def read_image(filename):
    return np.array(Image.open(filename.strip('\n')).convert('L'), np.uint8)


def read_file(filename, limit=0):
    n = 0
    lines = line_count(filename)

    with open(filename) as data:
        while True:
            line = next(data, None)
            if not line or (limit and n >= limit):
                break

            n += 1
            print '\r%s %d/%d' % (filename, n, limit or lines),
            try:
                yield read_image(line)
            except:
                continue


def get_templates():
    return np.array(list(read_file('templates.txt')))


def get_images(limit=0):
    positive = read_file('positive.txt', limit / 2 if limit else 0)
    negative = read_file('negative.txt', limit / 2 if limit else 0)

    for p, n in izip_longest(positive, negative):
        if p is not None:
            yield (1, p)
        if n is not None:
            yield (0, n)


def get_dataset(limit):
    return map(np.asarray, zip(*get_images(limit)))


def plot_roc(fpr, tpr, roc_auc):
    # Plot all ROC curves
    plt.figure()

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Affine Invariant SURF + Decision Tree Classifier')
    plt.legend(loc='lower right')
    plt.show()


def plot_importance(feature_count, importances, indices):
    plt.figure()
    plt.title('Feature importances')
    plt.bar(range(feature_count), importances[indices], color='r', align='center')
    plt.xticks(range(feature_count), indices)
    plt.xlim([-1, feature_count])
    plt.show()


def main(name, dataset_size):
    templates = get_templates()

    print 'templates: %d' % len(templates)

    labels, samples = get_dataset(dataset_size)

    print 'samples: %d' % len(samples)

    extractor = cv2.FeatureDetector_create('SURF')
    detector = cv2.DescriptorExtractor_create('SURF')

    print 'applying affine invariant transform'

    affine = AffineInvariant(extractor, detector)
    templates = affine.transform(templates)
    samples = affine.transform(samples)

    model = Pipeline([
        ('match', TemplateMatch(Templates(templates))),  # XXX: hack to bypass cloning error
        # ('reduce_dim', PCA(n_components = 12 * 6))
    ])

    samples = model.fit_transform(samples)

    rng = np.random.RandomState()

    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.5, random_state=rng)
    print 'train: %d, test: %d' % (len(X_train), len(X_test))

    params = dict(
        min_samples_split = [5, 6, 7, 8, 9, 10],
        min_samples_leaf = [3, 4, 5, 6, 7],
        max_leaf_nodes = [10, 9, 8, 7, 6],
        class_weight = [{1: w} for w in [10, 8, 4, 2, 1]]
    )

    tree = DecisionTreeClassifier(max_depth=4, random_state=rng)

    cvmodel = GridSearchCV(tree, params, cv=10, n_jobs=cv2.getNumberOfCPUs())
    cvmodel.fit(X_train, y_train)

    print 'grid scores'
    for params, mean_score, scores in cvmodel.grid_scores_:
        print '%0.3f (+/-%0.03f) for %r' % (mean_score, scores.std() * 2, params)
    print 'best parameters'
    print cvmodel.best_params_

    importances = cvmodel.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]

    plot_importance(6, importances, indices)

    y_pred = cvmodel.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print 'accuracy: %f' % accuracy
    print classification_report(y_test, y_pred)
    print confusion_matrix(y_test, y_pred)

    y_score = cvmodel.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plot_roc(fpr, tpr, roc_auc)

    export_graphviz(cvmodel.best_estimator_, out_file=name + '.dot', class_names=['background', 'badge'], filled=True, rounded=True, special_characters=True)
    pickle.dump(dict(params=params, pipe=model, model=cvmodel.best_estimator_), open(name + '.pkl', 'wb'))


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) >= 2 else 'classifier'
    dataset_size = int(sys.argv[2]) if len(sys.argv) >= 3 else 0

    main(name, dataset_size)
