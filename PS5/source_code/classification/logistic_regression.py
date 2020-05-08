
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import linear_model
from performance_utils import print_performance_metrics

def my_LogisticRegression(X_train,Y_train,X_test,Y_test):

    C = [0.01, 0.1, 1, 10]
    i = 1
    for param in C:
        log_reg = linear_model.LogisticRegression(C=param, solver='lbfgs')
        log_reg.fit(X_train, Y_train)
        Y_predict_test = log_reg.predict(X_test)
        y_predict_prob = log_reg.predict_proba(X_test)[:, 1]
        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(Y_test, y_predict_prob)
        plt.subplot(2, 2, i)
        plt.tight_layout()
        plt.plot(false_positive_rate, true_positive_rate, i)
        plt.xlim([-0.2, 1.2])
        plt.ylim([-0.2, 1.2])
        plt.title('ROC curve C = {}'.format(param))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        # Use score method to get accuracy of model
        score = log_reg.score(X_test, Y_test)
        print("+ C = %f => Test accuracy: %f" % (param, score))
        print_performance_metrics(Y_predict_test,Y_test)
        i = i + 1
    print("* Save Logistic Regression classification result into image/logistic_regression.png")
    plt.savefig('image/logistic_regression.png')
    plt.show()

    # define a function that accepts a threshold and prints sensitivity and specificity
    def evaluate_threshold(threshold):
        print('Sensitivity:', true_positive_rate[thresholds > threshold][-1])
        print('Specificity:', 1 - false_positive_rate[thresholds > threshold][-1])

    evaluate_threshold(0.9)
