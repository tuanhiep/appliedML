from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from performance_utils import print_performance_metrics

def my_SVM(X_train,Y_train,X_test,Y_test):
        # Support Vector Machine Classifier
        plt.clf()
        C = [0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, 20, 50]
        SVM_train_accuracy = []
        SVM_test_accuracy = []

        for param in C:
            clf = SVC(C=param, kernel='linear')
            clf.fit(X_train, Y_train)
            Y_predict_SVM_train = clf.predict(X_train)
            Y_predict_SVM_test = clf.predict(X_test)
            SVM_train_accuracy.append(accuracy_score(Y_train, Y_predict_SVM_train))
            SVM_test_accuracy.append(accuracy_score(Y_test, Y_predict_SVM_test))
            print("+ C = %f => Train accuracy = %f and Test accuracy = %f " % (
                param, accuracy_score(Y_train, Y_predict_SVM_train), accuracy_score(Y_test, Y_predict_SVM_test)))
            print_performance_metrics(Y_predict_SVM_test,Y_test)

        plt.plot(C, SVM_train_accuracy, 'ro-')
        plt.plot(C, SVM_test_accuracy, 'bv--')
        plt.legend(['Training Accuracy', 'Test Accuracy'])
        plt.xlabel('C')
        plt.xscale('log')
        plt.ylabel('Accuracy')
        plt.title('Support Vector Machine')
        print("* Save Support Vector Machine classification result into image/svm.png")
        plt.savefig('image/svm.png')
        plt.show()
