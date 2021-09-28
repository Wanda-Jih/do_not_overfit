# Challenge from Kaggle - Don't Overfit! ll

*This is the final project for machining learning course at [The Hong Kong Polytechnic University (PolyU)](https://www.polyu.edu.hk/en/)*

You can find this Kaggle challenge at this [link](https://www.kaggle.com/c/dont-overfit-ii/data)

### Goal
Predict the binary `target` associated with each row, without overfitting to the minimal set of training examples provided.

#### To solve the problem of overfitting
1. Increase the datasets(x)
1. Reduce the dimensions(ok)-features selection
1. Explicitly penalize overly complex models(ok)


### Work Flow
![](https://i.imgur.com/EfHjWcn.png)

### Datase
* train.csv - the training set. 250 rows.
* test.csv - the test set. 19,750 rows.
* sample_submission.csv - a sample submission file in the correct format

### Preprocessing


Using Pearson, RFECV, Probing to select features.
* Pearson's correlation : keep the positive correlation
* RFECV : uee Lasso to get features
* Probing : the most rude method, but it is intuitive to think of it. We train each column, and then test the trained model. If we can successfully test the testing set that means this column can be kept

The comparison is showing below. We can find that Probing can get the better result.
![](https://i.imgur.com/b0aeR84.png)




### Models
There are three models which are used in this project
1. SVC
2. Logistic Regression
3. Lasso 

Lasso model adds L1 penalty to standard linear model. Using the L1 penalty, the model can reduce some parameters to 0, which reduce the number of input dimension. So, this model is suitable for this dataset.

### Conclusion
We put our result to Kaggle, and get 0.871 score

