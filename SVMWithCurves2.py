import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
#from sklearn import svm
from sklearn.svm import SVC
randomSeed = 13  #Dataset1 - 11 through 15

MyFrame = pd.read_csv('C:/Users/User2/Desktop/GATECH/Machine Learning/Datasets/Dataset2 - Assignment 1/winequality-red.csv')
MyFrame['Quality'] = MyFrame['Quality'].apply(lambda x: 1 if x in (6,7,8) else 0) 
print(MyFrame.shape)

# Independent features and target feature
features = MyFrame.drop(columns=['Quality'])
target = MyFrame['Quality']
print(target.value_counts())

#Test Train Split: 75-25
X_train, X_test, Y_train, Y_test = train_test_split(features, target, random_state=randomSeed, test_size=0.25)
print("Shape of test set:", X_test.shape)
print("Shape of training set:", X_train.shape)

#Before Hyperparameter Tuning
MyTree = SVC(kernel="poly", degree=2, C=1, random_state=randomSeed)
MyTree.fit(X_train,Y_train)
predicting = MyTree.predict(X_test)
from sklearn.metrics import accuracy_score
score_test=accuracy_score(Y_test,predicting)
print(f'Accuracy before scaling or hyperparameter tuning: {score_test}')


#Scaling the train and test set
print(f'Scaling Begins...')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(f'Scaling Ends')

# Changing degree of poly kernel 
estimator = SVC(kernel="poly", C=1, gamma='auto',random_state=randomSeed)
train_score, test_score = validation_curve(
        estimator, X_train, Y_train,param_name= 'degree', param_range = range(1,7) , scoring='f1')

# Store stats for testing score 
mean_test_score = np.mean(test_score, axis = 1) 
std_test_score = np.std(test_score, axis = 1) 
# Store stats for training score 
mean_train_score = np.mean(train_score, axis = 1) 
std_train_score = np.std(train_score, axis = 1) 

# Plotting training and cross-validation scores 
param_range = range(1,7)
plt.plot(param_range, mean_train_score,  
     label = "Training Score", color = 'brown') 
plt.plot(param_range, mean_test_score, 
   label = "Cross-Validation Score", color = 'green') 

title = "Model Complexity/Validation Curve: Wine Quality Prediction"
plt.title(title)
plt.xlabel("Degree of Poly Kernel")
plt.ylabel("Model Score")
plt.axvline(x=param_range[np.argmax(mean_test_score)],color='r', linestyle='dotted')
plt.tight_layout() 
plt.legend(loc = 'best') 
plt.show()
print('Poly Degree for best accuracy: {}'.format(param_range[np.argmax(mean_test_score)]))

# Hyperparameter Tuning begins
# GridSearch for best parameters
parameters = {'kernel':['rbf', 'poly','sigmoid'], 'C':[0.01, 0.1, 1.0]} 
grid = GridSearchCV(SVC(degree=3, random_state=randomSeed), param_grid=parameters, scoring='f1',refit = True,cv=4, verbose = 10)
grid.fit(X_train, Y_train)
# Function Prints best parameters for GridSearchCV
print('Top Parameters: {}'.format(grid.best_params_))
print('Top Score: {}\n'.format(grid.best_score_))
print(f'This classification model has a top accuracy of {math.ceil(100*grid.best_score_)} % using scaling & grid search, we have an improvement over initial score of {math.floor(100*score_test)} % on the validation dataset\n')

# Learning Curve computation using the top parameters
val_set=[]
train_set=[]
new_dt = SVC(degree=3,**(grid.best_params_),random_state=randomSeed)
for itr in range(20,101,20):
    # Sampling Data
    df_samples = MyFrame.sample(frac=itr*0.01)
    X_train_samples=df_samples.drop('Quality',axis=1)
    Y_train_samples=df_samples['Quality']
    #Scale data sample
    X_train_samples=StandardScaler().fit_transform(X_train_samples)
    # Fit and Predict
    new_dt.fit(X_train_samples,Y_train_samples)
    predict_train = new_dt.predict(X_train_samples)
    predict_test = new_dt.predict(X_test)
    # F1-Accuracy score
    accr_train=accuracy_score(Y_train_samples,predict_train)
    accr_test=accuracy_score(Y_test,predict_test)
    train_set.append(accr_train)
    val_set.append(accr_test)
    print(f'Done for: {itr}% of total data')

# Plotting Learning Curve - initializing with taking 20% of data and going upto 100% training data
plt.plot(range(20,101,20),train_set,c='brown')
plt.plot(range(20,101,20),val_set,c='green')
title = "Learning Curve: Wine Quality Prediction"
plt.title(title)
plt.legend(["Training Scores","Validation Scores"])
plt.xlabel("Training Sample (in %)")
plt.ylabel("Model Accuracy")
plt.ylim(0.6,1.0)
plt.xticks(np.arange(0,120,step=20))
plt.show()
