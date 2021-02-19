# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Machine Learning: Assignment 1

# %%
#export
import numpy as np
import pandas as pd
import time
import gc
import random
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from sklearn.neural_network import MLPClassifier

# %% [markdown]
# # 1. Data Import and Cleansing Setup

# %%
class Data():
    def dataAllocation(self,path):
        df = pd.read_csv(path)
        x_data = df.iloc[:, :-1]
        y_data = df.iloc[:, -1 ]
        return x_data,y_data
    def trainSets(self,x_data,y_data):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 614, shuffle = True)
        return x_train, x_test, y_train, y_test

data = 'data/pima-indians-diabetes.csv'
dataset = Data()
x_data,y_data = dataset.dataAllocation(data)
x_train, x_test, y_train, y_test = dataset.trainSets(x_data,y_data)

# %% [markdown]
# # 2. Decision Tree Classifier

# %%
class DTClassifier():
    def decisionTreeClassifier(self,x_train,x_test, y_train):
        dt_clf = DecisionTreeClassifier(random_state = 614)
        dt_clf.fit(x_train, y_train)
        y_predict_train = dt_clf.predict(x_train)
        y_predict_test = dt_clf.predict(x_test)
        return dt_clf,y_predict_train, y_predict_test
    
    def dtTrainAccuracy(self,y_train,y_predict_train):
        return accuracy_score(y_predict_train, y_train)
    
    def dtTestAccuracy(self,y_test,y_predict_test):
        return accuracy_score(y_predict_test, y_test)

    def dtFeatureImportance(self,dt_clf):
        return dt_clf.feature_importances_

    def sortedDTFeatureImportanceIndicies(self,dt_clf):
        return np.argsort(dt_clf.feature_importances_, )

    def hyperParameterTuning(self,dt_clf,x_train,y_train):
        param_grid = {'max_depth': range(1, 20) }
        tune = GridSearchCV(estimator = dt_clf, param_grid = param_grid)
        return tune.fit(x_train, y_train)
    
    def bestParams(self,gscv_dtc):
        best_params = gscv_dtc.best_params_
        return best_params

    def bestScore(self,gscv_dtc):
        best_score = gscv_dtc.best_score_
        return best_score
    
dt = DTClassifier()
dt_clf,y_predict_train, y_predict_test = dt.decisionTreeClassifier(x_train,x_test, y_train)
print("Decision Tree Train Accuracy: ",dt.dtTrainAccuracy(y_train,y_predict_train))
print("Decision Tree Test Accuracy: ",dt.dtTestAccuracy(y_test,y_predict_test))
print("Decision Tree Feature Importance: ",dt.dtFeatureImportance(dt_clf))
print("Decision Tree Sorted Feature Importance: ",dt.sortedDTFeatureImportanceIndicies(dt_clf))
gscv_dtc = dt.hyperParameterTuning(dt_clf,x_train,y_train)
print("Decision Tree Best Parameters: ",dt.bestParams(gscv_dtc))
print("Decision Tree Best Score: ",dt.bestScore(gscv_dtc))

# %% [markdown]
# # 3. Support Vector Machine

# %%
class SupportVectorMachine():
    def dataPreProcess(self,x_train,x_test):
        scaler = StandardScaler()
        scaled_x_train = scaler.fit_transform(x_train)
        scaled_x_test = scaler.transform(x_test)
        return scaled_x_train, scaled_x_test

    def SVCClassifier(self,scaled_x_train,scaled_x_test, y_train):
        model = SVC(kernel = "linear")
        model.fit(scaled_x_train,y_train)
        y_predict_train = model.predict(scaled_x_train)
        y_predict_test = model.predict(scaled_x_test)
        return y_predict_train,y_predict_test

    def SVCTrainAccuracy(self,y_train,y_predict_train):
        train_accuracy = accuracy_score(y_predict_train, y_train)
        return train_accuracy

    def SVCTestAccuracy(self,y_test,y_predict_test):
        test_accuracy = accuracy_score(y_predict_test, y_test)
        return test_accuracy

    def SVMBestScore(self, scaled_x_train, y_train):
        svm_parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 0.1, 1.0]}
        svm_tune = SVC(gamma = "auto")
        svm_cv = GridSearchCV(estimator = svm_tune, param_grid = svm_parameters, n_jobs=-1, return_train_score=True)
        svm_cv.fit(scaled_x_train, y_train)
        best_score = svm_cv.best_score_
        print(svm_cv.best_params_)
        return svm_cv, best_score

    def SVCClassifierParam(self,svm_cv,scaled_x_train,scaled_x_test,y_train):
        y_predict_train = svm_cv.predict(scaled_x_train)
        y_predict_test = svm_cv.predict(scaled_x_test)
        return y_predict_train, y_predict_test

    def svcTrainAccuracy(self,y_train,y_predict_train):
        train_accuracy = accuracy_score(y_predict_train, y_train)
        return train_accuracy

    def svcTestAccuracy(self,y_test,y_predict_test):
        test_accuracy = accuracy_score(y_predict_test, y_test)
        return test_accuracy

    def SVMRankTestScore(self,svm_cv):
        rank_test_score = svm_cv.cv_results_['rank_test_score']
        return rank_test_score

    def SVMMeanTestScore(self,svm_cv):
        mean_test_score = svm_cv.cv_results_['mean_test_score']
        return mean_test_score

svm = SupportVectorMachine()
scaled_x_train, scaled_x_test = svm.dataPreProcess(x_train,x_test)
y_predict_train,y_predict_test = svm.SVCClassifier(scaled_x_train,scaled_x_test, y_train)
print("Support Vector Machine Trainn Accuracy: ",svm.SVCTrainAccuracy(y_train,y_predict_train))
print("Support Vector Machine Test Accuracy: ",svm.SVCTestAccuracy(y_test,y_predict_test))
svm_cv, best_score = svm.SVMBestScore(scaled_x_train, y_train)
print("Support Vector Machine Best Score: ", best_score)
y_predict_train,y_predict_test = svm.SVCClassifierParam(svm_cv,scaled_x_train,scaled_x_test,y_train)
print("Support Vector Machine Train Accuracy: ",svm.svcTrainAccuracy(y_train,y_predict_train))
print("Support Vector Machine Test Accuracy: ",svm.svcTestAccuracy(y_test,y_predict_test))
print("Support Vector Machine Rank Test Score: ",svm.SVMRankTestScore(svm_cv))
print("Support Vector Machine Mean Test Score: ",svm.SVMMeanTestScore(svm_cv))

# %% [markdown]
# # 4. KNN

# %%
class KNN():
    def kNeighborsClassifier(self,x_train,x_test, y_train, n_neighbors):
        dt_clf = KNeighborsClassifier(n_neighbors = n_neighbors)
        dt_clf.fit(x_train, y_train)
        y_predict_train = dt_clf.predict(x_train)
        y_predict_test = dt_clf.predict(x_test)
        return dt_clf,y_predict_train, y_predict_test
    
    def dtTrainAccuracy(self,y_train,y_predict_train):
        train_accuracy = accuracy_score(y_predict_train, y_train)
        return train_accuracy
    
    def dtTestAccuracy(self,y_test,y_predict_test):
        test_accuracy = accuracy_score(y_predict_test, y_test)
        return test_accuracy

    def hyperParameterTuning(self,x_train,y_train):
        train_scores, test_scores = validation_curve(KNeighborsClassifier(), x_train, y_train, param_name="n_neighbors", param_range=range(1, 50), cv=5)
        return np.argmax(np.mean(test_scores, axis=1)) + 1
    
    
dt = KNN()
dt_clf,y_predict_train, y_predict_test = dt.kNeighborsClassifier(x_train,x_test, y_train, 3)
print("Decision Tree Train Accuracy: ",dt.dtTrainAccuracy(y_train,y_predict_train))
print("Decision Tree Test Accuracy: ",dt.dtTestAccuracy(y_test,y_predict_test))
n_neighbors_tuned = dt.hyperParameterTuning(x_train,y_train)
print("Optimal n_neighbors: ", n_neighbors_tuned)
dt_clf,y_predict_train, y_predict_test = dt.kNeighborsClassifier(x_train,x_test, y_train, n_neighbors_tuned)
print("Decision Tree Train Accuracy: ",dt.dtTrainAccuracy(y_train,y_predict_train))
print("Decision Tree Test Accuracy: ",dt.dtTestAccuracy(y_test,y_predict_test))

# %% [markdown]
# # 5. Neural Network

# %%
class NN():
    def dataPreProcess(self,x_train,x_test):
        scaler = StandardScaler()
        scaled_x_train = scaler.fit_transform(x_train)
        scaled_x_test = scaler.transform(x_test)
        return scaled_x_train, scaled_x_test

    def mlpClassifier(self,scaled_x_train,scaled_x_test, y_train):
        model = MLPClassifier(hidden_layer_sizes=(8,8,8),activation='relu', solver='adam', max_iter=500)

        model.fit(scaled_x_train,y_train)
        y_predict_train = model.predict(scaled_x_train)
        y_predict_test = model.predict(scaled_x_test)
        return y_predict_train,y_predict_test

    def mlpTrainAccuracy(self,y_train,y_predict_train):
        train_accuracy = accuracy_score(y_predict_train, y_train)
        return train_accuracy

    def mlpTestAccuracy(self,y_test,y_predict_test):
        test_accuracy = accuracy_score(y_predict_test, y_test)
        return test_accuracy


svm = NN()
scaled_x_train, scaled_x_test = svm.dataPreProcess(x_train,x_test)
y_predict_train,y_predict_test = svm.mlpClassifier(scaled_x_train,scaled_x_test, y_train)
print("Neural Network Train Accuracy: ",svm.mlpTrainAccuracy(y_train,y_predict_train))
print("Neural Network Test Accuracy: ",svm.mlpTestAccuracy(y_test,y_predict_test))


# %%



