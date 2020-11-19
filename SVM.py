import numpy as np
from numpy import array
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
import tensorflow as tf
from sklearn.model_selection import train_test_split

num_votes = 12

results = []
best = 0

for vote_id in range(num_votes):
    print("VOTE = ", vote_id)

    # train label
    train_y = []
    read_label = open("/cluster_features/train_label.txt", "r")
    train_y = read_label.readlines()
    train_y = [int(i) for i in train_y]
    train_y = array(train_y)

    # test label
    y = []
    read_label = open("/cluster_features/label.txt", "r")
    y = read_label.readlines()
    y = [int(i) for i in y]
    y = array(y)

    # train featrue
    train_X = []
    read_feature = open("/cluster_features/train_feature_"+str(vote_id)+".txt", "r")
    lines = read_feature.readlines()
    for line in lines:
        line = line.split('\n')[0]
        line = line.split(' ')
        # print(line)
        line = [float(i) for i in line]
        train_X.append(line)
    train_X = array(train_X)

    # test featrue
    X = []
    read_feature = open("/cluster_features/feature_"+str(vote_id)+".txt", "r")
    lines = read_feature.readlines()
    for line in lines:
        line = line.split('\n')[0]
        line = line.split(' ')
        line = [float(i) for i in line]
        X.append(line)
    X = array(X)

    # Change percentage of training samples
    #X_drop, train_X, y_drop, train_y = train_test_split(train_X, train_y, test_size = 0.01)


    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    clf.fit(train_X, train_y)

    if clf.score(X, y) > best:
        best = clf.score(X, y)
    results.append(clf.score(X, y))
    print(clf.score(X, y))
    pred = clf.predict(X)
    print(tf.math.confusion_matrix(y, pred).numpy())

results = array(results)
print('best', best)
print('mean', np.mean(results))
