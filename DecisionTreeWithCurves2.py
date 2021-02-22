import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
import copy
import sklearn.tree as tree
randomSeed = 11  #Dataset1 - 11 through 15

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

#Before Hyperparameter Tuning (No pruning)
from sklearn.tree import DecisionTreeClassifier
MyTree = DecisionTreeClassifier(random_state=randomSeed)
MyTree.fit(X_train,Y_train)
predicting = MyTree.predict(X_test)
from sklearn.metrics import accuracy_score
score_test=accuracy_score(Y_test,predicting)
print(f'Accuracy before hyperparameter tuning: {score_test}')

#With Pruning - Changing max_depth parameter 
estimator = DecisionTreeClassifier(random_state=randomSeed) 
train_score, test_score = validation_curve(
        estimator, X_train, Y_train,param_name= 'max_depth', param_range = np.arange(1, 11) ,cv = 4, n_jobs=1, scoring='f1')

# Store stats for testing score 
mean_test_score = np.mean(test_score, axis = 1) 
std_test_score = np.std(test_score, axis = 1) 
# Store stats for training score 
mean_train_score = np.mean(train_score, axis = 1) 
std_train_score = np.std(train_score, axis = 1) 
  
# Plotting Model Complexity Curve by changing max_depth
param_range = np.arange(1, 11)
plt.plot(param_range, mean_train_score,  
     label = "Training Score", color = 'brown') 
plt.plot(param_range, mean_test_score, 
   label = "Cross-Validation Score", color = 'green') 
title = "Model Complexity/Validation Curve: Wine Quality Prediction"
plt.title(title)
plt.xlabel("Max_depth")
plt.ylabel("Model Score")
plt.axvline(x=param_range[np.argmax(mean_test_score)],color='r', linestyle='dotted')
plt.tight_layout() 
plt.legend(loc = 'best') 
plt.show()

# Hyperparameter Tuning begins
# GridSearch for best parameters
parameters = {
    'max_depth': list(range(1, 11))
    ,'criterion':['gini','entropy'] # we can comment it out if it takes more time, it will NOT change our scores significantly
    ,'min_samples_leaf': list(range(1, 6))
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=randomSeed), param_grid=parameters, cv=4, n_jobs=-1, 
                      verbose=10, return_train_score=True, scoring='f1')
grid.fit(X_train, Y_train)
# Function Prints best parameters for GridSearchCV
print('Top Parameters: {}'.format(grid.best_params_))
print('Top Score: {}\n'.format(grid.best_score_))
print(f'This classification model has a top accuracy of {math.ceil(100*grid.best_score_)} % using grid search which is an improvement over initial score of {math.floor(100*score_test)} % on the validation dataset\n')

# Pruning Result
Complete_dt = copy.deepcopy(grid.best_estimator_)
Complete_dt.max_depth = 10 
Complete_dt.fit(X_train, Y_train)
print(f'The entire tree has {Complete_dt.tree_.node_count} nodes')
Pruned_dt = grid.best_estimator_
print(f'The pruned tree has {Pruned_dt.tree_.node_count} nodes only')

# Learning Curve computation using the top parameters
val_set=[]
train_set=[]
new_dt = DecisionTreeClassifier(random_state=randomSeed,**(grid.best_params_))
for itr in range(20,101,20):
    # Sampling Data
    df_samples = MyFrame.sample(frac=itr*0.01)
    X_train_samples=df_samples.drop('Quality',axis=1)
    Y_train_samples=df_samples['Quality']
    # Fit and Predict
    new_dt.fit(X_train_samples,Y_train_samples)
    predict_train = new_dt.predict(X_train_samples)
    predict_test = new_dt.predict(X_test)
    # Accuracy score
    accr_train=accuracy_score(Y_train_samples,predict_train)
    accr_test=accuracy_score(Y_test,predict_test)
    train_set.append(accr_train)
    val_set.append(accr_test)
    print(f'Done for: {itr}% of total data')
# Plotting Learning Curve
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

