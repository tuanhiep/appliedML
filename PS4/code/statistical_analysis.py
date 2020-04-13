from pandas.api.types import is_numeric_dtype

def myStatistical_analysis(data2):
    print('#----------------------------- Statistical Analysis  -------------------------------------#')
    for col in data2.columns:
        if is_numeric_dtype(data2[col]):
            print('%s:' % (col))
            print('\t Mean = %.2f' % data2[col].mean())
            print('\t Standard deviation = %.2f' % data2[col].std())
            print('\t Minimum = %.2f' % data2[col].min())
            print('\t Maximum = %.2f' % data2[col].max())
    #Data Covariance:
    print("Covariance: ",data2.cov())
    #Data Correlation:
    print("\n\nCorrelation: ",data2.corr())


