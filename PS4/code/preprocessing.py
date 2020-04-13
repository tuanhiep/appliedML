import numpy as np

def myPreprocessing(data):
    
    print('#--------------------- Preprocessing ----------------------------------#')
          
    print(data)
# The original data has some mistype value, replace
    data.replace('ckd\t', 'ckd', regex=True)
    # Make sure that there are only 2
    print(data['classification'].value_counts())
    # Replace missing data
    data = data.replace('?',np.NaN)
    
    print('Number of instances = %d' % (data.shape[0]))
    print('Number of attributes = %d' % (data.shape[1]))
    
    print('Number of missing values:')
    for col in data.columns:
        print('\t%s: %d' % (col,data[col].isna().sum()))
    
    # Drop the missing data
    print('Number of rows in original data = %d' % (data.shape[0]))
    data2 = data.dropna()
    print('Number of rows after discarding missing values = %d' % (data2.shape[0]))
    
    # Make the text data become numeric
    data2 = data2.replace('normal',1)
    data2 = data2.replace('abnormal',0)
    data2 = data2.replace('present',1)
    data2 = data2.replace('notpresent',0)
    data2 = data2.replace('no',0)
    data2 = data2.replace('yes',1)
    data2 = data2.replace('poor',0)
    data2 = data2.replace('good',1)
    data2 = data2.replace('notckd',0)
    data2 = data2.replace('ckd',1)
    
    
    # Save data to a csv file
    np.savetxt('csv/kidneyclean.csv', data2, delimiter=',', fmt='%s')
    
    return data2