from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from performance_utils import print_performance_metrics

def my_NonLinearSVM(X_train,Y_train,X_test,Y_test):

    C = [0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, 20, 50]
    NL_SVM_train_Accuracy = []
    NL_SVM_test_Accuracy = []

    for param in C:
        clf = SVC(C=param, kernel='rbf', gamma='auto')
        clf.fit(X_train, Y_train)
        Y_predict_NL_SVMTrain = clf.predict(X_train)
        Y_predict_NL_SVM_Test = clf.predict(X_test)
        NL_SVM_train_Accuracy.append(accuracy_score(Y_train, Y_predict_NL_SVMTrain))
        NL_SVM_test_Accuracy.append(accuracy_score(Y_test, Y_predict_NL_SVM_Test))
        print(" * C = %f => Train accuracy = %f and Test accuracy = %f " % (
            param, accuracy_score(Y_train, Y_predict_NL_SVMTrain), accuracy_score(Y_test, Y_predict_NL_SVM_Test)))
        print_performance_metrics(Y_predict_NL_SVM_Test,Y_test)
    plt.plot(C, NL_SVM_train_Accuracy, 'ro-')
    plt.plot(C, NL_SVM_test_Accuracy, 'bv--')
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.xlabel('C')
    plt.xscale('log')
    plt.ylabel('Accuracy')
    plt.title('Nonlinear Support Vector Machine')
    print("* Save Non Linear Support Vector Machine classification result into image/non_linear_svm.png")
    plt.savefig('image/non_linear_svm.png')
    plt.show()
