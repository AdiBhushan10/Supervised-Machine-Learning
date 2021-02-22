import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.neural_network import MLPClassifier
randomSeed = 15  #Dataset1 - 11 through 15

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
MyTree = MLPClassifier(random_state=randomSeed)
MyTree.fit(X_train,Y_train)
predicting = MyTree.predict(X_test)
from sklearn.metrics import accuracy_score
score_test=accuracy_score(Y_test,predicting)
print(f'Accuracy before hyperparameter tuning: {score_test}')

#Changing hidden_layer_sizes parameter 
estimator = MLPClassifier(random_state=randomSeed,solver='adam',activation='logistic') 
train_score, test_score = validation_curve(
        estimator, X_train, Y_train,param_name= 'hidden_layer_sizes', param_range = range(10,101,10) ,scoring='f1')
mean_test_score = np.mean(test_score, axis = 1) 
mean_train_score = np.mean(train_score, axis = 1)   
# Plotting training and cross-validation scores 
param_range =  range(10,101,10)
plt.plot(param_range, mean_train_score,  
     label = "Training Score", color = 'brown') 
plt.plot(param_range, mean_test_score, 
   label = "Cross-Validation Score", color = 'green') 
title = "Model Complexity/Validation Curve: Wine Quality Prediction"
plt.title(title)
plt.xlabel("Hidden Layers")
plt.ylabel("Model Score")
plt.axvline(x=param_range[np.argmax(mean_test_score)],color='r', linestyle='dotted')
plt.tight_layout() 
plt.legend(loc = 'best') 
plt.show()
print('hidden_layer_sizes for best accuracy: {}'.format(param_range[np.argmax(mean_test_score)]))

# Hyperparameter Tuning begins
# GridSearch for best parameters
parameters = {'hidden_layer_sizes': [80], 'learning_rate_init': [0.05, 0.01, 0.001,0.1],'activation': ['identity', 'logistic','tanh','relu']}
grid = GridSearchCV(MLPClassifier(random_state=randomSeed), param_grid=parameters, cv=4, 
                      verbose=10, return_train_score=True, scoring='f1')
grid.fit(X_train, Y_train)
# Function Prints best parameters for GridSearchCV
print('Top Parameters: {}'.format(grid.best_params_))
print('Top Score: {}\n'.format(grid.best_score_))
print(f'This classification model has a top accuracy of {math.ceil(100*grid.best_score_)} % using grid search which is an improvement over initial score of {math.floor(100*score_test)} % on the validation dataset\n')

# Learning Curve computation using the top parameters
val_set=[]
train_set=[]
new_dt = MLPClassifier(random_state=randomSeed,**(grid.best_params_))
for itr in range(20,101,20):     
    # Sampling Data
    df_samples = MyFrame.sample(frac=itr*0.01)
    # Generate target
    X_train_samples=df_samples.drop('Quality', axis=1)
    Y_train_samples=df_samples['Quality']
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
plt.ylim(0.0,1.0)
plt.xticks(np.arange(0,120,step=20))
plt.show()


# Plotting Learning Curve - varying the trainings with ephocs from 1 to 5
val_set=[]
train_set=[]
for i in range(1,500,100):
    # Classifier Initialized     
    new_dt = MLPClassifier(random_state=randomSeed,**(grid.best_params_),solver='adam', max_iter = i) 
    # Fit and Predict
    new_dt.fit(X_train,Y_train)
    predict_train = new_dt.predict(X_train)
    predict_test = new_dt.predict(X_test)
    # F1-Accuracy score
    accr_train=accuracy_score(Y_train,predict_train)
    accr_test=accuracy_score(Y_test,predict_test)
    train_set.append(accr_train)
    val_set.append(accr_test)
    print(f'Done for: {i} epoch')

# Plotting Learning Curve - initializing with taking 20% of data and going upto 100% training data
plt.plot(range(1,500,100),train_set,c='brown')
plt.plot(range(1,500,100),val_set,c='blue')
title = "Learning Curve wrt Epochs: Wine Quality Prediction"
plt.title(title)
plt.legend(["Training Scores","Validation Scores"])
plt.xlabel("Number of Epochs")
plt.ylabel("Model Accuracy")
plt.ylim(0.0,1.0)
plt.xticks(np.arange(0,500,step=50))
plt.show()
