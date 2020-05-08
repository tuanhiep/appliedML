import numpy as np
from pandas.api.types import is_numeric_dtype

def my_statistical_analysis(data):
    # for col in data.columns:
    #     if is_numeric_dtype(data[col]):
    #         print('%s:' % (col))
    #         print('\t Mean = %.2f' % data[col].mean())
    #         print('\t Standard deviation = %.2f' % data[col].std())
    #         print('\t Minimum = %.2f' % data[col].min())
    #         print('\t Maximum = %.2f' % data[col].max())
    #Data Covariance:
    print("* Save the covariance into csv/covariance.csv")
    data.cov().to_csv("csv/covariance.csv")
    #Data Correlation:
    print("* Save the correlation into csv/correlation.csv")
    data.corr().to_csv("csv/correlation.csv")
