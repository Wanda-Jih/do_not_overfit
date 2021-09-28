import os
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso

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
    data = pd.read_csv(path, index_col=0) #去掉第一行的預設值
    testing = data[features]
    return testing
     
def lasso(x_train,y_train): 

    lasso = Lasso(
            alpha=1.0, 
            copy_X=True, 
            fit_intercept=True, 
            max_iter=1000,
            normalize=False, 
            positive=False, 
            precompute=False, 
            random_state=None,
            selection='cyclic', 
            tol=0.0001, 
            warm_start=False)

    lasso.fit(x_train, y_train) 
    return lasso

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

    # svc model
    lasso_model = lasso(x_train, y_train)
    
    # predict the result
    predict(lasso_model, test)




