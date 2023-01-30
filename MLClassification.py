# import comet_ml at the top of your file
from comet_ml import Experiment
from neo4j import GraphDatabase
import pandas as pd
import dateutil.parser
from datetime import datetime
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import xgboost as xgb
import numpy as np
from statistics import mean
from sklearn.metrics import classification_report
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import statistics
from sklearn.decomposition import SparsePCA 
from collections import Counter
from sklearn.datasets import make_classification
from numpy import where
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from imblearn.pipeline import make_pipeline
from imblearn.combine import SMOTEENN
from xgboost import plot_importance
from sklearn.model_selection import StratifiedKFold
from xgboost import cv
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import svm
from sklearn import metrics
from dataProcessing import getPreprocessData
from sklearn.metrics import roc_curve, roc_auc_score
import time

import warnings
from functools import reduce
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib
from sklearn.linear_model import LassoCV

# Create an experiment with your api key
experiment = Experiment(
    api_key="yO67iXhjD5FRQH0uKNVOkpCuq",
    project_name="MasterThesis",
    workspace="aftab571",
)

st_time_nodes = time.time()
df_admission,df_labs, df_vitals,df_output = getPreprocessData('ML',edge_merge=False,grp_aggr='mean')


df_admission = df_admission.iloc[:,df_admission.columns.isin(['label','hadm_id'])]
df_labs = df_labs.iloc[:,~df_labs.columns.isin(['label'])]

def merge_data(df_labs,df_vitals,df_output):
    new_df = reduce(lambda x,y: pd.merge(x,y, on='hadm_id', how='left'), [df_admission,df_labs, df_vitals, df_output])
    return new_df

df= merge_data(df_labs,df_vitals,df_output)
#df = df.dropna(subset=df.columns.to_list(), thresh=5)
df= df.drop(['marital','ethnicity','religion','gender'],axis=1)

# df.to_csv('check.csv')
# df = pd.read_csv('check.csv')
# df_nan = pd.DataFrame()
# df_nan['val']=df.notna().sum()
# df_nan.sort_values(by='val', ascending=True)
# nan_dict= df_nan[df_nan['val']>4000].to_dict()
# nan_list= list(nan_dict['val'].keys())
# df= df.iloc[:,df.columns.isin(nan_list)]

# df=df.drop(['Unnamed: 0'],axis=1)
end_time = time.time()
df = df.reset_index(drop=True)
print("Time for Preprocessing data (ML): ",end_time-st_time_nodes)
df['label'] = df['label'].astype(int)
X = df.iloc[:,~df.columns.isin(['label','hadm_id'])]
Y = df['label']

data_dmatrix = xgb.DMatrix(data=X,label=Y)

X_train_complete, X_test, y_train_complete, y_test = train_test_split(X, Y, test_size=0.30,random_state=10,stratify=Y)

params = {
            'objective':'binary:logistic',
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':100
        }   
xg_cl = xgb.XGBClassifier(**params)

svm_clf = svm.SVC(kernel='linear')

skf = StratifiedKFold(n_splits=2, random_state=10, shuffle=True)
lst_accu_stratified_val = []

for train_index, test_index in skf.split(X_train_complete, y_train_complete):
    x_train_fold, x_test_fold = X_train_complete.iloc[train_index], X_train_complete.iloc[test_index]
    y_train_fold, y_test_fold = y_train_complete.iloc[train_index], y_train_complete.iloc[test_index]
    #x_train_fold= x_train_fold.fillna(0)
#     print(x_train_fold.head(5))
#     pipeline = make_pipeline(SimpleImputer(),MinMaxScaler(), SMOTE())
#     x_train_fold,y_train_fold = pipeline.fit_resample(x_train_fold, y_train_fold)
#     print(x_train_fold.head(5))

#     counter = Counter(y_train_fold)
#     print(counter)
#     # transform the dataset
#     oversample = SMOTE()
#     x_train_fold, y_train_fold = oversample.fit_resample(x_train_fold, y_train_fold)
#     # # summarize the new class distribution
#     counter = Counter(y_train_fold)
#     print(counter)

    smote = SMOTE(k_neighbors=4, n_jobs=-1)
    smote_enn = make_pipeline(SimpleImputer(), SMOTEENN(smote=smote))  #SMOTEENN(smote=smote)
    #cols_arr = x_train_fold.columns.to_list()
    x_train_fold,y_train_fold = smote_enn.fit_resample(x_train_fold, y_train_fold)
    #x_train_fold = pd.DataFrame(x_train_fold, columns=cols_arr)



    # Fit the classifier to the training set
    xg_cl.fit(x_train_fold, y_train_fold)
    svm_clf.fit(x_train_fold, y_train_fold)
    
    # Predict the labels of the test set: preds
    XG_y_pred_val = xg_cl.predict(x_test_fold)


    # Compute the accuracy: accuracy validation
    accuracy_val = float(np.sum(XG_y_pred_val == y_test_fold)) / y_test_fold.shape[0]
    lst_accu_stratified_val.append(accuracy_val)
    
    #print(classification_report(y_test_fold, XG_y_pred_val, target_names=['0','1']))
    print("XGBoost Validation accuracy: %f" % (accuracy_val))

     # Predict the labels of the test set: preds
    SVM_y_pred_val = xg_cl.predict(x_test_fold)


    # Compute the accuracy: accuracy validation
    accuracy_val = float(np.sum(SVM_y_pred_val == y_test_fold)) / y_test_fold.shape[0]
    lst_accu_stratified_val.append(accuracy_val)
    
    #print(classification_report(y_test_fold, SVM_y_pred_val, target_names=['0','1']))
    print("SVM Validation accuracy: %f" % (accuracy_val))
print(lst_accu_stratified_val)
X_test = X_test.fillna(0)

XG_y_pred = xg_cl.predict(X_test)
accuracy = float(np.sum(XG_y_pred == y_test)) / y_test.shape[0]
print(classification_report(y_test, XG_y_pred, target_names=['0','1']))
print("XGBoost Test accuracy: %f" % (accuracy))
experiment.log_metric("XGBoost Test accuracy", accuracy)

svm_y_pred = svm_clf.predict(X_test)
accuracy = float(np.sum(svm_y_pred == y_test)) / y_test.shape[0]
print(classification_report(y_test, svm_y_pred, target_names=['0','1']))
print("SVM Test accuracy: %f" % (accuracy))

experiment.log_metric("SVM Test accuracy:", accuracy)


experiment.log_confusion_matrix(y_test, XG_y_pred,title="XGBoost")
experiment.log_confusion_matrix(y_test, svm_y_pred,title="SVM")



plot_importance(xg_cl,max_num_features=20)
plt.savefig("XGBImp.png", dpi=400)
#plt.show()




XG_cf_matrix = confusion_matrix(y_test, XG_y_pred)
SVM_cf_matrix = confusion_matrix(y_test, svm_y_pred)

print("XGBoost Test: ",XG_cf_matrix)

sensitivity1 = XG_cf_matrix[0,0]/(XG_cf_matrix[0,0]+XG_cf_matrix[0,1])
print('Sensitivity : ', sensitivity1 )
experiment.log_metric("XGBoost Sensitivity", sensitivity1)

specificity1 = XG_cf_matrix[1,1]/(XG_cf_matrix[1,0]+XG_cf_matrix[1,1])
print('Specificity : ', specificity1)
experiment.log_metric("XGBoost Specificity", specificity1)

print("SVM Test: ",SVM_cf_matrix)

sensitivity1 = SVM_cf_matrix[0,0]/(SVM_cf_matrix[0,0]+SVM_cf_matrix[0,1])
print('Sensitivity : ', sensitivity1 )
experiment.log_metric("SVM Sensitivity", sensitivity1)

specificity1 = SVM_cf_matrix[1,1]/(SVM_cf_matrix[1,0]+SVM_cf_matrix[1,1])
print('Specificity : ', specificity1)
experiment.log_metric("SVM Specificity", specificity1)

ax = sns.heatmap(XG_cf_matrix, annot=True, cmap='Blues', fmt= '.3g')

ax.set_title('XGBoost Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Survived','Died'])
ax.yaxis.set_ticklabels(['Survived','Died'])

## Display the visualization of the Confusion Matrix.
plt.savefig('XGBoost_conf.png', dpi=400)
plt.show()


ax1 = sns.heatmap(SVM_cf_matrix, annot=True, cmap='Blues', fmt= '.3g')

ax1.set_title('SVM Confusion Matrix\n\n')
ax1.set_xlabel('\nPredicted Values')
ax1.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax1.xaxis.set_ticklabels(['Survived','Died'])
ax1.yaxis.set_ticklabels(['Survived','Died'])

## Display the visualization of the Confusion Matrix.
plt.savefig('SVMConf.png', dpi=400)
plt.show()



#set up plotting area
plt.figure(0).clf()


fpr, tpr, _ = metrics.roc_curve(y_test, XG_y_pred)
auc = round(metrics.roc_auc_score(y_test, XG_y_pred), 4)
plt.plot(fpr,tpr,label="XGBoost, AUC="+str(auc))


fpr, tpr, _ = metrics.roc_curve(y_test, svm_y_pred)
auc = round(metrics.roc_auc_score(y_test, svm_y_pred), 4)
plt.plot(fpr,tpr,label="SVM, AUC="+str(auc))
plt.legend()
plt.savefig('SVM_XGB_AUC.png', dpi=400)
#add legend

plt.show()

#X = df.iloc[:,df.columns != 'expire']
#Y = df['expire']
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20,random_state=1,stratify=Y)
cols_arr = X.columns.to_list()
smote = SMOTE(k_neighbors=5, n_jobs=-1)
smote_enn = make_pipeline(SimpleImputer(), SMOTEENN(smote=smote))
X,Y = smote_enn.fit_resample(X, Y)
X = pd.DataFrame(X, columns=cols_arr)
reg = LassoCV()
reg.fit(X, Y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,Y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.savefig("fImp.png", dpi=400)




