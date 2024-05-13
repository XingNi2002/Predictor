# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pickle


# 加载.mat数据集
def load_mat_dataset(file_path):
    data_with_indices = scipy.io.loadmat(file_path)['data_with_indices']
    # 提取特征值、标签和交叉验证折数
    X = data_with_indices[:, :-2]
    y = data_with_indices[:, -2].ravel()
    folds = data_with_indices[:, -1].ravel()
    return X, y, folds


# 加载数据集
features, labels, folds = load_mat_dataset("./Data/DR1_smote_ENN.mat")

# 定义分类器100、90、110
rfc1 = RandomForestClassifier(n_estimators=170)
rfc2 = RandomForestClassifier(n_estimators=200)
gbc = GradientBoostingClassifier(n_estimators=580)

# 定义集成模型
voting_classifier = VotingClassifier(estimators=[('rfc1', rfc1), ('rfc2', rfc2), ('gbc', gbc)], voting='soft')

# 初始化十折交叉验证
kf = KFold(n_splits=10)

# 初始化性能指标列表
accuracies = []

# 十折交叉验证
for train_index, test_index in kf.split(features):
    # 划分训练集和测试集
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # 训练集成模型
    voting_classifier.fit(X_train, y_train)

    # 在测试集上进行预测
    predictions = voting_classifier.predict(X_test)

    # 计算模型准确率
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)

# 输出十折交叉验证的性能指标
print("十折交叉验证的性能指标:", accuracies)
print("平均准确率:", np.mean(accuracies))

# 保存模型到文件
with open("new_ensemble.pkl", "wb") as model_file:
    pickle.dump(voting_classifier, model_file)

