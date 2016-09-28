import cv2
import sys
import pickle

from AffineInvariantFeatures import AffineInvariant
from RegionOfInterest import extractRoi
from TemplateMatcher import TemplateMatch, Templates #imported to allow deserialization

from matplotlib import pyplot as plt

def labelSamples(coords, rect):
    pass

def main():
    image = cv2.imread(sys.argv[1])
    winSize = (256, 256)
    stepSize = 64

    samples = [(rect, cv2.cvtColor(window, cv2.COLOR_BGR2GRAY))
               for rect, window in extractRoi(image, winSize, stepSize)]

    X_test = [window for rect, window in samples]
    coords = [rect for rect, window in samples]
    labels = labelSamples(coords, rect)

    extractor = cv2.FeatureDetector_create('SURF')
    detector = cv2.DescriptorExtractor_create('SURF')

    affine = AffineInvariant(extractor, detector)

    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size = 0.5, random_state = rng)
    print 'train: %d, test: %d' % (len(X_train), len(X_test))

    params = dict(
        min_samples_split = [5, 6, 7, 8, 9, 10],
        min_samples_leaf = [3, 4, 5, 6, 7],
        max_leaf_nodes = [10, 9, 8, 7, 6],
        class_weight = [{1: w} for w in [10, 8, 4, 2, 1]]
    )

    tree = DecisionTreeClassifier(max_depth = 4, random_state = rng)

    cvmodel = GridSearchCV(tree, params, cv = 10, n_jobs = cv2.getNumberOfCPUs())
    cvmodel.fit(X_train, y_train)

    print 'best parameters'
    print cvmodel.best_params_
    print 'grid scores'
    for params, mean_score, scores in cvmodel.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)

    importances = cvmodel.best_estimator_.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in cvmodel.best_estimator_.estimators_], axis = 0)
    indices = np.argsort(importances)[::-1]

    plot_importance(6, importances, indices)#, std)

    y_pred = cvmodel.predict(X_test)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print accuracy
    print classification_report(y_test, y_pred)
    print confusion_matrix(y_test, y_pred)

    for i, result in enumerate(y_pred):
        if not y_test[i] and y_test[i] != result:
            index = np.where(np.all(samples == X_test[i], axis = 1))
            worst_samples[index[0][0]] += 1

    if best_accuracy[0] < recall:
        best_accuracy = (recall, accuracy, n)

    y_score = cvmodel.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plot_roc(fpr, tpr, roc_auc)

    name = 'match-dt-%d' % n
    export_graphviz(cvmodel.best_estimator_, out_file = name + '.dot', class_names = ['background', 'badge'], filled = True, rounded = True, special_characters = True)
    pickle.dump(dict(params = params, pipe = model, model = cvmodel.best_estimator_), open(name + '.pkl', 'wb'))

if __name__ == '__main__':
    main()
