import pandas as pd
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NearMiss



#Gaussian Oversampling
def Gaussian_Oversample(input_file, output_file):
    # 读取数据集
    data = pd.read_csv(input_file)
    X = data.iloc[:, :-1]  # 特征矩阵
    y = data.iloc[:, -1]  # 标签向量

    # 计算正负样本数量
    positive_count = sum(y == 1)  # 计算正样本数量
    negative_count = sum(y == 0)  # 计算负样本数量
    target_count = (positive_count + negative_count) // 2  # 目标的合成正样本数量

    # 生成服从正态分布的样本
    np.random.seed(0)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    num_samples_to_generate = target_count - positive_count
    synthetic_samples = pd.DataFrame(np.random.normal(loc=mean, scale=std, size=(num_samples_to_generate, X.shape[1])), columns=X.columns)
    synthetic_labels = pd.Series([1] * num_samples_to_generate)

    # 合并原始样本和生成的合成样本
    balanced_X = pd.concat([X, synthetic_samples], axis=0)
    balanced_y = pd.concat([y, synthetic_labels], axis=0)

    # 将过采样后的数据集保存到本地
    balanced_data = pd.concat([balanced_X, balanced_y], axis=1)
    balanced_data.to_csv(output_file, index=False)



#ADASYN Oversampling
def ADASYN_Oversample(input_file, output_file):
    # 读取数据集
    data = pd.read_csv(input_file)
    X = data.iloc[:, :-1]  # 特征矩阵
    y = data.iloc[:, -1]  # 标签向量

    # 计算正负样本数量
    positive_count = sum(y == 1)  # 计算正样本数量
    negative_count = sum(y == 0)  # 计算负样本数量
    target_count = (positive_count + negative_count) // 2  # 目标的合成正样本数量

    # 使用 ADASYN 过采样算法
    adasyn = ADASYN(sampling_strategy={1: target_count})  # 1表示正样本的标签
    balanced_X, balanced_y = adasyn.fit_resample(X, y)

    # 将过采样后的数据集保存到本地
    balanced_data = pd.concat([balanced_X, balanced_y], axis=1)
    balanced_data.to_csv(output_file, index=False)  # 保存为 CSV 文件


#Borderline Oversampling
def Borderline_Oversample(input_file, output_file):
    # 读取数据集
    data = pd.read_csv(input_file)
    X = data.iloc[:, :-1]  # 特征矩阵
    y = data.iloc[:, -1]  # 标签向量

    # 计算正负样本数量
    positive_count = sum(y == 1)  # 计算正样本数量
    negative_count = sum(y == 0)  # 计算负样本数量
    target_count = (positive_count + negative_count) // 2  # 目标的合成正样本数量

    # 使用 Borderline SMOTE 过采样算法
    smote = BorderlineSMOTE(sampling_strategy={1: target_count})  # 1表示正样本的标签
    balanced_X, balanced_y = smote.fit_resample(X, y)

    # 将过采样后的数据集保存到本地
    balanced_data = pd.concat([balanced_X, balanced_y], axis=1)
    balanced_data.to_csv(output_file, index=False)  # 保存为 CSV 文件


#SMOTE Oversampling
def SMOTE_Oversample(input_file, output_file):
    # 读取数据集
    data = pd.read_csv(input_file)
    X = data.iloc[:, :-1]  # 特征矩阵
    y = data.iloc[:, -1]  # 标签向量

    # 计算正负样本数量
    positive_count = sum(y == 1)  # 计算正样本数量
    negative_count = sum(y == 0)  # 计算负样本数量
    target_count = (positive_count + negative_count) // 2  # 目标的合成正样本数量

    # 使用 SMOTE 过采样算法
    smote = SMOTE(sampling_strategy={1: target_count})  # 1表示正样本的标签
    balanced_X, balanced_y = smote.fit_resample(X, y)

    # 将过采样后的数据集保存到本地
    balanced_data = pd.concat([balanced_X, balanced_y], axis=1)
    #balanced_data.to_csv(output_file, index=False)  # 保存为 CSV 文件
    # 将过采样后的数据集保存到本地，并添加一列标记为3
    balanced_data_with_label_3 = pd.concat([balanced_data, pd.Series([3] * len(balanced_data), name='Label_3')], axis=1)

    # 将合成的数据集保存到本地
    balanced_data_with_label_3.to_csv(output_file, index=False)

#ENN Undersampling
def ENN_Undersample(input_file, output_file):
    # 读取数据集
    data = pd.read_csv(input_file)

    # 提取特征和标签
    X = data.iloc[:, :-1]  # 特征
    y = data.iloc[:, -1]  # 标签

    # 使用 ENN 欠采样
    enn = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=12)  # 设置 sampling_strategy='auto' 表示自动调整样本数量
    X_resampled, y_resampled = enn.fit_resample(X, y)

    # 输出欠采样后的正负样本个数
    print("欠采样后的正样本个数：", sum(y_resampled == 1))
    print("欠采样后的负样本个数：", sum(y_resampled == 0))

    # 将欠采样后的数据保存到新文件
    resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=['label'])], axis=1)
    resampled_data.to_csv(output_file, index=False)

#Near Miss Undersampling
def Near_Undersample(input_file, output_file):
    # 读取CSV数据集
    data = pd.read_csv(input_file)

    # 分离特征和标签
    X = data.drop('label', axis=1)
    y = data['label']

    # 使用NearMiss欠采样方法
    nm = NearMiss()
    X_resampled, y_resampled = nm.fit_resample(X, y)

    # 将欠采样后的数据保存为新的CSV文件
    resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
    resampled_data.to_csv(output_file, index=False)



SMOTE_Oversample('./Data/InitialData/AAC_data.csv','1.csv')