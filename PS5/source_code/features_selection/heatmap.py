import matplotlib.pyplot as plt
import seaborn as sns

def my_heatmap(data2,name):
    corrmat = data2.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    #plot heat map
    sns.heatmap(data2[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    print("* Save the heatmap of "+name+" features into "+"image/heatmap_corelations_"+name+".png")
    plt.savefig("image/heatmap_corelations_"+name+".png")
    plt.clf()
