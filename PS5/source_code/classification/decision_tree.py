#from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
import pydotplus
from sklearn import tree
from performance_utils import print_performance_metrics
def my_DecisionTree(X_train,X_test,y_train,y_test):
    print('* Use Entropy index for impurity measure :')
    accuracy = np.empty(2, dtype=float)
    max_depths = [2, 3]
    i = 0
    for max_depth in max_depths:
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
        clf = clf.fit(X_train, y_train)
        # create graph tree with class names
        dot_data = tree.export_graphviz(clf, feature_names=X_train.columns, class_names=['1', '0'], filled=True,
                                        out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png('image/tree_entropy_%d.png' % (max_depth))
        predY = clf.predict(X_test)
        accuracy[i] = accuracy_score(y_test, predY)
        print('+ Entropy: Max depth %d , Accuracy on test data is %.2f' % (max_depth, (accuracy[i])))
        print_performance_metrics(predY,y_test)
        i += 1
    # plt.clf()
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].set_prop_cycle(color=['red'])
    ax[0].set_ylim([0.8, 1.1])
    ax[0].plot(max_depths, accuracy)
    ax[0].legend(['accuracy-Entropy'], loc='upper left')

    #  Using Gini index as impurity measure, fit decision trees of different maximum depths [2, 3, 4, 5,
    #  6, 7, 8, 9, 10, 15, 20, 25] to the training set
    i = 0
    print('* Use Gini index for impurity measure :')
    for max_depth in [2, 3]:
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth)
        clf = clf.fit(X_train, y_train)
        dot_data = tree.export_graphviz(clf, feature_names=X_train.columns, class_names=['1', '0'],
                                        filled=True,
                                        out_file=None)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_png('image/tree_gini_%d.png' % (max_depth))
        predY = clf.predict(X_test)
        accuracy[i] = accuracy_score(y_test, predY)
        print('+ Gini: Max depth %d , Accuracy on test data is %.2f' % (max_depth, (accuracy_score(y_test, predY))))
        print_performance_metrics(predY,y_test)
        i += 1

    ax[1].set_prop_cycle(color=['green'])
    ax[1].set_ylim([0.8, 1.1])
    ax[1].plot(max_depths, accuracy)
    ax[1].legend(['accuracy-Gini'], loc='upper left')
    plt.title('Decision Tree')
    print("* Save Decision Tree classification result into image/decision_tree.png")
    plt.savefig('image/decision_tree.png')
    plt.show()
