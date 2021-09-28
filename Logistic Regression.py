import os
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

data_dir= 'data/'

# probing
features=['0','9','15','16','17','24','33','39','43','45','63','65','73','80','89','90','91','98','101',
 '105','117','133','134','143','150','156','164','176','183','189','194','199','209','214','215',
 '217','221','227','228','230','237','239','240','244','253','258','276','295','298']


def get_train_data():
    path = os.path.join(data_dir,'train.csv')
    data = pd.read_csv(path, index_col=0)  
    x_train = data.iloc[:,1:] 
    y_train = data['target']
    return x_train, y_train 

def get_test_data():
    path = os.path.join(data_dir, 'test.csv')
    data = pd.read_csv(path, index_col=0)
    testing = data[features]
    return testing
     
def log(x_train,y_train): 
    log = LogisticRegression(
            penalty='l1',
            dual=False,
            tol=1e-4,
            C=0.2, 
            fit_intercept=False, 
            intercept_scaling=1, 
            class_weight='balanced',
            random_state=None,
            solver='liblinear', 
            multi_class= 'ovr', 
            verbose=0, 
            warm_start=False, 
            n_jobs=1)
           
    log.fit(x_train,y_train) 
    return log

def predict(model, test):
    prediction = model.predict(test)
    prediction = pd.DataFrame(prediction)
    prediction.index += 250
    prediction.columns = ['target']
    # prediction.to_csv('result/svc.csv', index_label='id', index=True)

if __name__ == "__main__":
    
    # get training and testing data
    x_train0, y_train = get_train_data()
    test = get_test_data()
    
    # preprocessing data
    scaler = RobustScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train0),columns=x_train0.columns,index=x_train0.index)
    test = pd.DataFrame(scaler.fit_transform(test),columns=test.columns,index=test.index)

    # logistic regression model
    log_model = log(x_train, y_train)
    
    # predict the result
    predict(log_model, test)

