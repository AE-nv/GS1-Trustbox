#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: karel.symoens
"""

# Packages
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime

# Read the data data ---------------------------------------------------------------------------------------------------
path = os.getcwd() + '/data/recp_Complete_TB_download_in_Excel_1462/'
all_files = glob.glob(path + "/*.xlsx")
excel_list = []
for filename in all_files[0:1]:
    one_excel = pd.read_excel(filename, index_col=None, header=list(range(1,7)), dtype=str)
    excel_list.append(one_excel)
raw_start_df = pd.concat(excel_list, axis=0, ignore_index=True)

raw_header_lines = pd.read_excel(
        os.getcwd() + '/data/recp_Complete_TB_download_in_Excel_1462/recp_Complete_TB_download_in_Excel_14620.xlsx'
        , header=None
        , sheet_name='Sheet0'
        , nrows=7)

#create headers
raw_header_lines.replace('empty|HEADER',np.nan,regex=True,inplace=True)
#n_clmn = range(0,raw_header_lines.shape[1])
headers=list()
for col in range(raw_header_lines.shape[1]):
    column_data = raw_header_lines.iloc[:,col]
    headers.append(column_data.str.cat(sep='.'))
raw_start_df.columns=headers
start_data = raw_start_df.iloc[8:]
start_data.reset_index(drop=True, inplace=True)

#select a subset of the data and simplify headers
allergen_data = start_data[['productData.gtin',
                       'basicProductInformationModule.productName',
                       'basicProductInformationModule.productName/@languageCode',
                       'productAllergenInformationModule.allergenRelatedInformation.allergen.allergenTypeCode',
                       'productAllergenInformationModule.allergenRelatedInformation.allergen.levelOfContainmentCode',
                       'foodAndBeverageIngredientInformationModule.ingredientStatement',
                       'foodAndBeverageIngredientInformationModule.ingredientStatement/@languageCode']]
allergen_data.loc[:,'productData.gtin'].fillna(method='ffill', inplace=True)
allergen_data.columns = ['gtin', 'productName', 'productName.langCode', 'allergenTypeCode', 
                         'levelOfContainmentCode', 'ingredientStatement', 
                         'ingredientStatement.langCode']
allergen_data.loc[allergen_data['levelOfContainmentCode']=='FREE_FROM','allergenTypeCode']=np.nan
allergen_data = allergen_data[~((allergen_data['productName'].isna()) & 
                                (allergen_data['productName.langCode'].isna()) & 
                                (allergen_data['allergenTypeCode'].isna()) & 
                                (allergen_data['ingredientStatement'].isna()) & 
                                (allergen_data['ingredientStatement.langCode'].isna()))] #remove rows with NaNs

#import manually determined language codes
languageDetectionData_manual_adaptations = pd.read_excel(r'/Users/karel.symoens/languageDetectionData_manual_adaptations.xlsx')
real_languages = languageDetectionData_manual_adaptations[['ID', 'manualLangCode']]
real_languages.set_index('ID',inplace=True)
allergen_data_wRL = pd.concat([allergen_data,real_languages], axis=1, join='outer')

#use if you want to work with manual language codes
allergen_data['ingredientStatement.langCode'] = allergen_data_wRL['manualLangCode']

#create product name columns
pivot_prodName = allergen_data.pivot(columns='productName.langCode', values='productName')
pivot_prodName_subset = pivot_prodName[['de','en','fr','nl']]
pivot_prodName_subset.columns=['de.prodName', 'en.prodName', 'fr.prodName', 'nl.prodName']

#create ingredient list columns
pivot_ingrStat = allergen_data.pivot(columns='ingredientStatement.langCode', values='ingredientStatement')
pivot_ingrStat_subset = pivot_ingrStat[['de','en','fr','nl']]
pivot_ingrStat_subset.columns=['de.ingrStat', 'en.ingrStat', 'fr.ingrStat', 'nl.ingrStat']

#create allergen dummies
allergens_dummies = allergen_data['allergenTypeCode'].str.get_dummies()

#merge everything
data_f = pd.concat([allergen_data['gtin'], 
                    pivot_prodName_subset, 
                    pivot_ingrStat_subset, 
                    allergens_dummies],axis=1)

#fill nan
data_f['en.prodName'] = data_f.groupby('gtin')['en.prodName'].apply(lambda x: x.ffill().bfill())
data_f['nl.prodName'] = data_f.groupby('gtin')['nl.prodName'].apply(lambda x: x.ffill().bfill())
data_f['fr.prodName'] = data_f.groupby('gtin')['fr.prodName'].apply(lambda x: x.ffill().bfill())
data_f['de.prodName'] = data_f.groupby('gtin')['de.prodName'].apply(lambda x: x.ffill().bfill())

data_f['en.ingrStat'] = data_f.groupby('gtin')['en.ingrStat'].apply(lambda x: x.ffill().bfill())
data_f['nl.ingrStat'] = data_f.groupby('gtin')['nl.ingrStat'].apply(lambda x: x.ffill().bfill())
data_f['fr.ingrStat'] = data_f.groupby('gtin')['fr.ingrStat'].apply(lambda x: x.ffill().bfill())
data_f['de.ingrStat'] = data_f.groupby('gtin')['de.ingrStat'].apply(lambda x: x.ffill().bfill())

#data_f[['en.prodName', 'nl.prodName', 'fr.prodName', 'de.prodName']] = data_f[['en.prodName', 'nl.prodName',
#     'fr.prodName', 'de.prodName']].fillna('Not Available')
#data_f[['en.ingrStat', 'nl.ingrStat', 'fr.ingrStat', 'de.ingrStat']] = data_f[['en.ingrStat', 'nl.ingrStat',
#     'fr.ingrStat', 'de.ingrStat']].fillna('Not Available')

#aggregate per gtin
aggregated_per_product = data_f.groupby('gtin')['en.prodName','fr.prodName', 'nl.prodName', 
                                       'de.ingrStat','en.ingrStat', 'fr.ingrStat', 'nl.ingrStat'].first()
allergen_indicator_cols = data_f.columns[9:]
aggregated_allergen_indicators = data_f.groupby('gtin')[allergen_indicator_cols].sum()
aggregated_allergen_booleans=aggregated_allergen_indicators>=1
aggregated_data=pd.concat([aggregated_per_product, aggregated_allergen_booleans['AM']], axis=1)

#Predictive Model -------------------------------------------------------------
#packages
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
nltk.download('punkt')
 
#remove products with no IngrStat
data = aggregated_data[aggregated_data['en.ingrStat'].notnull()]

#create tokens & list of ingredients
listOfIngrLists=[]
for ingrList in data['en.ingrStat']:
    ingrList_wo = re.sub(r"[^a-zA-Z]+", ' ', ingrList).strip()
    lowerIngrList = ingrList_wo.lower()
    lowerIngrList = lowerIngrList.split(' ')
    tmplist=[]
    for ingredient in lowerIngrList:
        if ingredient not in stopwords:
            tmplist.append(ingredient)
    tmplist2=[]
    for ingredient in tmplist:
        tmplist2.append(ps.stem(ingredient))
    unique_ingr = set(tmplist2)
    listOfIngrLists.append(unique_ingr)
data['en.ingrList']=listOfIngrLists

#dummies
mlb = MultiLabelBinarizer()
ingrDummies = pd.DataFrame(mlb.fit_transform(data['en.ingrList']),columns=mlb.classes_, index=data.index)

#filter ingredients that appear only once
freq_bool = ingrDummies.sum()>1
limited_ingrDummies = ingrDummies[freq_bool.index[freq_bool]]
subset_data = data[['en.prodName','en.ingrStat','en.ingrList']]
clean_df = pd.concat([subset_data, limited_ingrDummies], axis=1)

#Create train and test set
np.random.seed(14)
X_train, X_test, y_train, y_test = train_test_split(limited_ingrDummies, data['AM'], test_size=0.2,random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#train RF
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(X_train,y_train)

#test performance
predictions = clf.predict(X_test)
X_test['model_prediction'] = predictions

#AUC
fpr, tpr, thresholds = metrics.roc_curve(y_true = y_test, y_score = predictions, pos_label=1)
metrics.auc(fpr, tpr)

#feature importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = limited_ingrDummies.columns
names = [features[i] for i in indices]
imp_value = [importances[i] for i in indices]
feature_imp = pd.DataFrame(list(zip(names, imp_value)))

#important_features = feature_imp[feature_imp[1]!=0]
important_features = feature_imp.iloc[0:200,0]

#rf limited features
lim_clf = RandomForestClassifier(n_estimators=1000,
                                 n_jobs=2, 
                                 random_state=0,
                                 max_depth=50)
lim_clf.fit(X_train[important_features], y_train)

#test performance
predictions_limf = lim_clf.predict(X_test[important_features])
X_test['model_prediction_lim_features'] = predictions_limf

#AUC
fpr2, tpr2, thresholds = metrics.roc_curve(y_true = y_test, y_score = predictions_limf, pos_label=1)
metrics.auc(fpr2, tpr2)

#accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions_limf)

#playing with parameters
from sklearn.model_selection import GridSearchCV
param_grid = {  
           "n_estimators" : [10, 100, 500, 750, 1000, 1250],  
           "max_depth" : [5, 50, 100, 250, 500],  
           "min_samples_leaf" : [1, 2, 3, 5, 10, 20, 40]}  

cv_rf = GridSearchCV(estimator=lim_clf, param_grid=param_grid, scoring='accuracy')  
cv_rf.fit(X_train, y_train)  
print(cv_rf.best_params_) 

# manual labelling of test set
subset_data_for_for_manual = data[['nl.prodName','en.ingrStat', 'AM']]
excel_for_manual_AM_labels = subset_data_for_for_manual.merge(X_test['model_prediction_lim_features'],left_index=True, right_index=True)
am_runtime = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
path = '/Users/karel.symoens/excel_for_manual_AM_labels_{}.xlsx'
excel_for_manual_AM_labels.to_excel(path.format(am_runtime), index_label='gtin')

#read manual enriched test data
manually_enriched_data = pd.read_excel('/Users/karel.symoens/excel_for_manual_AM_labels_2019-11-18_13:46:37 - handmatige controle.xlsx',
                                       index_col=0)
#accuracy
from sklearn.metrics import accuracy_score
handmatige_resultaten = X_test.merge(manually_enriched_data['handmatige_AM'], left_index=True, right_index=True)
accuracy_score(handmatige_resultaten['handmatige_AM'], handmatige_resultaten['model_prediction_lim_features'])
