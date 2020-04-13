
from sklearn.decomposition import PCA

def myPCA(X):
    print('#----------------------------- Feature Extraction : PCA -------------------------------------#')
    
    pca = PCA(n_components=8)
    fit = pca.fit(X)
    # summarize components
    print("Explained Variance: %s" % fit.explained_variance_ratio_)
    print(fit.components_)