from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import pandas as pd

def myExtraTreesClassifier(X,Y):
    print('#----------------------------- Extra Trees Classifier -------------------------------------#')
    
    model = ExtraTreesClassifier()
    model.fit(X,Y)
    print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.xlabel('Importance Scores')
    # naming the y axis
    plt.ylabel('Attributes')
    
    # giving a title to my graph
    plt.title('Feature Importance')
    plt.savefig("image/featureimportance.png")
    plt.clf()