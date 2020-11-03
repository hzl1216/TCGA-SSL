#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import preprocessing  
from collections import Counter
def feature_selection_and_sort_by_chromosome(data, annotation_path):
    big_table = pd.read_csv(data)
    feature_name = list(big_table)
    feature_name = feature_name[1:]
    labels = np.array(big_table.iloc[:, 0])
    annotation = pd.read_csv(annotation_path, dtype=str)
    gene_id_annotation = list(annotation.loc[:, "gene"])
    feature_name =[feature.split('|')[0] for feature in feature_name]
    idx = []
    print('features have been sorted based on chromosome')
    for gene_id in gene_id_annotation:
        if gene_id in feature_name:
            idx.append(feature_name.index(gene_id))
    feature_name=np.array(feature_name)[idx]
    features_raw = np.array(big_table.iloc[:, 1:])
    features = np.log2(1.0 + features_raw)
    features[np.where(features <= 1)] = 0
    # numpy is different from lis
    features = features[:, idx]
    print('remove the features that  Variance is low than threshold')
    selector = VarianceThreshold(threshold=1)
    selector.fit(features)
    idx2 = selector.get_support(indices=True)
    features = features[:, idx2]
    feature_name=np.array(feature_name)[idx2]

    kf= KFold(n_splits=5, shuffle=True, random_state=13)
    for index, (train_index, test_index) in enumerate(kf.split(features)):
        x_train, x_test, y_train,y_test = features[train_index],features[test_index],labels[train_index],labels[test_index]
        scaler = normalise_and_save(x_train,feature_name,y_train ,file_path='data/train_%d.csv'%index)
        normalise_and_save(x_test,feature_name,y_test,scaler,'data/test_%d.csv'%index)

def merge_geo_tcga(tcga_data, geo_data, annotation_path):
    tcga_table = pd.read_csv(tcga_data)
    geo_table = pd.read_csv(geo_data)
    feature_name_t = list(tcga_table)[1:]
    feature_name_t = [feature.split('|')[0] for feature in feature_name_t]
    feature_name_g = list(geo_table)[1:]
    labels = np.array(tcga_table.iloc[:, 0])
    labels = np.array(tcga_table.iloc[:, 0])
    annotation = pd.read_csv(annotation_path, dtype=str)
    gene_id_annotation = list(annotation.loc[:, "gene"])
    idx_t = []
    idx_g = []
    print('features have been sorted based on chromosome')
    for gene_id in gene_id_annotation:
        if gene_id in feature_name_t and gene_id in feature_name_g:
            idx_t.append(feature_name_t.index(gene_id))
            idx_g.append(feature_name_g.index(gene_id))

    feature_name = np.array(feature_name_t)[idx_t]

    features_t = np.array(tcga_table.iloc[:, 1:])
    features_t = np.log2(1.0 + features_t)
    features_t[np.where(features_t <= 1)] = 0
    features_t = features_t[:, idx_t]

    features_g = np.array(geo_table.iloc[:, 1:])
    features_g[np.where(features_g <= 1)] = 0
    features_g = features_g[:, idx_g]
    # numpy is different from lis

    print('remove the features that  Variance is low than threshold')
    selector = VarianceThreshold(threshold=1)
    selector.fit(features_t)
    idx2 = selector.get_support(indices=True)
    features_t = features_t[:, idx2]
    features_g = features_g[:, idx2]
    feature_name = np.array(feature_name)[idx2]

    kf = KFold(n_splits=5, shuffle=True, random_state=13)
    scaler = preprocessing.MinMaxScaler()
    features_t = scaler.fit_transform(features_t)

    for index, (train_index, test_index) in enumerate(kf.split(features_t)):
        x_train, x_test, y_train, y_test = features_t[train_index], features_t[test_index],labels[train_index], labels[test_index]
        normalise_and_save(x_train, feature_name, y_train, scaler, 'data/train_%d.csv'%index)
        normalise_and_save(x_test, feature_name, y_test, scaler, 'data/test_%d.csv'%index)
    features_g = scaler.transform(features_g)
    pd.DataFrame(features_g, columns=feature_name).to_csv('data/geo_data.csv', index=0)

def normalise_and_save(features,feature_name,labels,scaler=None,file_path='train.csv'):
    print(features.shape,len(feature_name))
    print('normalise the data in [0,1])')

    if scaler is None:
        scaler = preprocessing.MinMaxScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    feature_name_path = os.path.join(file_path)
    features = np.concatenate((labels.reshape(-1,1),features),axis=1)
    feature_name = np.concatenate((np.array(['label']),feature_name))
    print(features.shape,feature_name.shape)
    pd.DataFrame(features,columns=feature_name).to_csv(feature_name_path,index=0)
    print('features are selected, the selected preprocessing data are saved at', feature_name_path)
    return scaler


if __name__ == '__main__':
    # feature_selection_and_sort_by_chromosome('data/TCGA_data.csv','data/Annotation.csv')
    merge_geo_tcga('D:/TCGA-DATA/big_gene_expression_data.csv','D:/GEO数据/big_data_xiong.csv','D:/TCGA-DATA/Annotation.csv')

    '''
    big_table1 = pd.read_csv('D:/GEO数据/big_data_xiong.csv',nrows =5)
    big_table2 = pd.read_csv('D:/TCGA-DATA/big_gene_expression_data.csv',nrows =5)
    annotation = pd.read_csv('D:/TCGA-DATA/Annotation.csv', dtype=str)
    gene_id_annotation = list(annotation.loc[:, "gene"])
    print(gene_id_annotation)
    feature_name1 = list(big_table1)
    feature_name2 = list(big_table2)
    feature_name2 = [feature.split('|')[0] for feature in feature_name2]
    for gene_id in feature_name2:
        if gene_id in gene_id_annotation:
            print(gene_id)
    print(feature_name2)
    count = 0
    for i in feature_name1:
        if i in feature_name2:
            count += 1
    print(count)
    '''







