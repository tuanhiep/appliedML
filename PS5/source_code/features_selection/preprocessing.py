import numpy as np
import pandas as pd
def myPreprocessing(data):

    print("* Print out original data for visualization :")
    print(data.head())
    # data.drop(data.columns[0], axis=1)
    # The original data has some mistype value, replace
    data.replace('ckd\t', 'ckd', regex=True)
    # Make sure that there are only 2
    print("* Print out original labels for visualization :")
    print(data['classification'].value_counts())
    # Replace missing data
    data = data.replace('?',np.NaN)
    print('* Number of instances = %d' % (data.shape[0]))
    print('* Number of attributes = %d' % (data.shape[1]-2))
    print('* Number of missing values:')
    for col in data.columns:
        print('\t%s: %d' % (col,data[col].isna().sum()))

    # Drop the missing data
    print('* Number of rows in original data = %d' % (data.shape[0]))
    print("* Drop missing values")
    preprocessed_data = data.dropna()
    print('* Number of rows after discarding missing values = %d' % (preprocessed_data.shape[0]))

    # Make the text data become numeric
    preprocessed_data = preprocessed_data.replace('normal',1)
    preprocessed_data = preprocessed_data.replace('abnormal',0)
    preprocessed_data = preprocessed_data.replace('present',1)
    preprocessed_data = preprocessed_data.replace('notpresent',0)
    preprocessed_data = preprocessed_data.replace('no',0)
    preprocessed_data = preprocessed_data.replace('yes',1)
    preprocessed_data = preprocessed_data.replace('poor',0)
    preprocessed_data = preprocessed_data.replace('good',1)
    preprocessed_data = preprocessed_data.replace('notckd',0)
    preprocessed_data = preprocessed_data.replace('ckd',1)
    preprocessed_data = preprocessed_data.apply(pd.to_numeric)

    # Save data to a csv file
    print("* Save preprocessed data into csv/kidneyclean.csv")
    np.savetxt('csv/kidneyclean.csv', preprocessed_data, delimiter=',', fmt='%s')

    return preprocessed_data
