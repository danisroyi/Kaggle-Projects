# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 08:07:07 2018

@author: danis

Create functions 
that will make the kaggle submission faster and easier
1. functions for data transforms
2. function for model run
3. function for kaggle submission
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import warnings


pd.set_option('display.max_columns',7)
pd.set_option('display.width',300)


warnings.filterwarnings('ignore')

def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

def process_fare(df,cut_points,label_names):
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean())
    df['Fare_categories'] = pd.cut(df['Fare'],cut_points,labels=label_names)
    return df

def process_name(df,titles):
    extracted_titles = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
    df["Title"] = extracted_titles.map(titles)
    return df

def process_cabin(df):
    df["Cabin_type"] = df["Cabin"].str[0]
    df["Cabin_type"] = df["Cabin_type"].fillna("Unknown")
    df["Cabin_type"] = df["Cabin_type"].str.replace('T','Unknown')
    return df
  
def process_pclass(df):
   df['Pclass_cat']  = df['Pclass'].astype('category') 
   return df

def process_embarked(df):
    df['Embarked'] = df['Embarked'].fillna('S')
    return df


def process_family(df):
    df['FamilySize'] = df.Parch + df.SibSp + 1
    df['BigFamily'] = df.FamilySize.apply(lambda s: s if s < 5 else 5)
    df['BigFamily_cat']  = df['BigFamily'].astype('category')
    return df

def process_ticket_count(df,cut_points,label_names):
    # select females and masters (boys)
    boy = (df.Name.str.contains('Master')) | ((df.Sex=='male') & (unified_df.Age<13))
    female = df.Sex=='female'
    boy_or_female = boy | female
    # no. females + boys on ticket that survived
    n_ticket = df[boy_or_female].groupby('Ticket').Survived.count()
    # survival rate amongst females + boys on ticket
    #tick_surv = unified_df[boy_or_female].groupby('Ticket').Survived.mean()
    df["ticket_num"] = pd.cut(n_ticket,cut_points,labels=label_names)
    #df["ticket_surv"] = pd.cut(tick_surv,cut_points,labels=label_names)
    return df

def process_numeric_cols(df,numeric_cols):
    for col in numeric_cols:
        df[col + '_scaled'] = minmax_scale(df[col])
    return df
    

def process_categorical_cols(df,cat_cols):
    df_features = df[cat_cols]
    dummy_df = pd.get_dummies(df_features[cat_cols])
    df_features = pd.concat([df_features,dummy_df],axis=1)
    df_features =  df_features.drop(cat_cols,axis=1)
    return df_features


def run_model_train_test(train_X, test_X, train_y, test_y,model):
    model.fit(train_X,train_y)
    predictions = model.predict(test_X)
    return predictions

def run_model_cv(model,df_feat,target,kf_num):
    kf = KFold(kf_num, shuffle=True, random_state=0)
    mses = cross_val_score(model,df_feat,target, cv=kf)
    avg_rmse = np.mean(mses)
    return mses,avg_rmse


def run_model_for_kaggle(model,train_feat,train_target,holdout):
    model.fit(train_feat,train_target)
    predictions = model.predict(holdout)
    return predictions

def prepare_for_kaggle_csv(pass_id,predictions):
    kaggle_df = pd.concat([pass_id,predictions],axis=1)
    kaggle_df.to_csv('C:\\Python\\Exercise\\Files\\Kaggle\\titanic_submission.csv',index=False)


train = pd.read_csv('C:\\Python\\Exercise\\Files\\titanic_train.csv')
holdout =  pd.read_csv('C:\\Python\\Exercise\\Files\\titanic_test.csv')
#keep the passenger_id for kaggle submission
passenger_id = holdout['PassengerId']

#combine to one df in order to simplify data transformations
unified_df = pd.concat([train,holdout],sort=False)


print()
print(len(unified_df['Ticket'].unique()))

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing",'Infant',"Child",'Teenager','Young Adult',"Adult",'Senior']
unified_df = process_age(unified_df,cut_points,label_names)

cut_points = [0,12,50,100,1000]
label_names = ['0-12','12-50','50-100','100+']
unified_df = process_fare(unified_df,cut_points,label_names)

cut_points = [0,1,3,5,10]
label_names = ['0-1','1-3','3-5','5+']
unified_df = process_ticket_count(unified_df,cut_points,label_names)

titles = {
    "Mr" :         "Mr",
    "Mme":         "Mrs",
    "Ms":          "Mrs",
    "Mrs" :        "Mrs",
    "Master" :     "Master",
    "Mlle":        "Miss",
    "Miss" :       "Miss",
    "Capt":        "Officer",
    "Col":         "Officer",
    "Major":       "Officer",
    "Dr":          "Officer",
    "Rev":         "Officer",
    "Jonkheer":    "Royalty",
    "Don":         "Royalty",
    "Sir" :        "Royalty",
    "Countess":    "Royalty",
    "Dona":        "Royalty",
    "Lady" :       "Royalty"
}

unified_df = process_embarked(unified_df)
unified_df = process_name(unified_df,titles)
unified_df = process_family(unified_df)
unified_df = process_cabin(unified_df)
unified_df = process_pclass(unified_df)
numeric_cols = ['SibSp','Parch','Fare']
unified_df = process_numeric_cols(unified_df,numeric_cols)


f_cols = ['Pclass_cat','Sex','Age_categories','Embarked','Title','Cabin_type']
feat_df = process_categorical_cols(unified_df,f_cols)


buf_df = pd.concat([feat_df,unified_df.filter(like='_scaled')],axis=1)
buf_df = pd.concat([buf_df,unified_df['Survived']],axis=1)

#split back to train and holdout before starting the model runs 
train = buf_df[buf_df['Survived'].isna() != True]
train['Survived'] = train['Survived'].astype('int64')
holdout =  buf_df[buf_df['Survived'].isna()]

print()
print(holdout.info())

train_target = train['Survived']
train = train.drop(['Survived'],axis=1)
holdout = holdout.drop(['Survived'],axis=1)

train_X, test_X, train_y, test_y = train_test_split(
    train,train_target, test_size=0.2,random_state=0)

#prepare to enesmble the results of all models with VotingClassifier
enesmblers = []

lr = LogisticRegression()
enesmblers.append(('lr',lr)) # add to enesmble each model as a list of tuple pairs
predictions = run_model_train_test(train_X, test_X, train_y, test_y,lr)
accuracy = accuracy_score(test_y, predictions)
print()
print('lr accuracy:')
print(accuracy)

mses,avg_rmse_lr = run_model_cv(lr,train,train_target,10)
print()
print(mses)
print(avg_rmse_lr)

rf = RandomForestClassifier(n_estimators=15,random_state=1,\
                             min_samples_leaf=5)
enesmblers.append(('rf',rf))
predictions = run_model_train_test(train_X, test_X, train_y, test_y,rf)
accuracy = accuracy_score(test_y, predictions)
print()
print('rf accuracy:')
print(accuracy)


mses,avg_rmse_rf = run_model_cv(rf,train,train_target,10)
print()
print(mses)
print(avg_rmse_rf)

knn = KNeighborsClassifier(n_neighbors=5)
#enesmblers.append(('knn',knn))
predictions = run_model_train_test(train_X, test_X, train_y, test_y,knn)
accuracy = accuracy_score(test_y, predictions)
print()
print('knn accuracy:')
print(accuracy)

mses,avg_rmse_knn = run_model_cv(knn,train,train_target,10)
print()
print(mses)
print(avg_rmse_knn)

xgb = XGBClassifier(objective= 'binary:logistic', eval_metric="auc", min_samples_leaf=5)
#enesmblers.append(('xgb',xgb))
predictions = run_model_train_test(train_X, test_X, train_y, test_y,xgb)
accuracy = accuracy_score(test_y, predictions)
print()
print('xgb accuracy:')
print(accuracy)

#mses,avg_rmse = run_model_cv(xgb,train,train_target,10)
#print()
#print(mses)
#print(avg_rmse)

svm = SVC(kernel='linear',gamma=1,C=50)
#enesmblers.append(('svm',svm))
predictions = run_model_train_test(train_X, test_X, train_y, test_y,svm)
accuracy = accuracy_score(test_y, predictions)
print()
print('svm accuracy:')
print(accuracy)

mses,avg_rmse_svm = run_model_cv(svm,train,train_target,10)
print()
print(mses)
print(avg_rmse_svm)




# create the ensemble model with all defaults of VotingClassifier
ene = VotingClassifier(enesmblers)
mses,avg_rmse_ene = run_model_cv(ene,train,train_target,10)
print()
print(mses)
print(avg_rmse_ene)



lr_pred = run_model_for_kaggle(lr,train,train_target,holdout)
rf_pred = run_model_for_kaggle(rf,train,train_target,holdout)
knn_pred = run_model_for_kaggle(knn,train,train_target,holdout)
ene_pred =  run_model_for_kaggle(ene,train,train_target,holdout)

#send to kaggle the results of the best model according to average 
#cross validation score
model_dict = {'lr': avg_rmse_lr,'rf': avg_rmse_rf,\
              'knn': avg_rmse_knn, 'ene':avg_rmse_ene}


best_model = max(model_dict, key=model_dict.get)

if best_model == 'lr':
    prepare_for_kaggle_csv(passenger_id,pd.Series(data=lr_pred,name='Survived'))
    print()
    print('best model is lr')
elif best_model == 'rf':
    prepare_for_kaggle_csv(passenger_id,pd.Series(data=rf_pred,name='Survived'))
    print()
    print('best model is rf')
elif best_model == 'knn':
    prepare_for_kaggle_csv(passenger_id,pd.Series(data=knn_pred,name='Survived'))
    print()
    print('best model is knn')
else:
    prepare_for_kaggle_csv(passenger_id,pd.Series(data=ene_pred,name='Survived'))
    print()
    print('best model is ene') #enesmbler improved accuracy, chosen as best
    








