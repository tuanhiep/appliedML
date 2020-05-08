from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import pandas as pd

def my_ExtraTreesClassifier(X,Y):

    model = ExtraTreesClassifier()
    model.fit(X,Y)
    #print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    print("* Print out 10 best features")
    print(feat_importances.nlargest(10))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.xlabel('Importance Scores')
    # naming the y axis
    plt.ylabel('Attributes')

    # giving a title to my graph
    plt.title('Feature Importance')
    print("* Save 10 most important features into image/extra_trees_classifier_feature_importance.png ")
    plt.savefig("image/extra_trees_classifier_feature_importance.png")
    plt.clf()
