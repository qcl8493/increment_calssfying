from scipy.stats import skew, kurtosis
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from features_time_fre import time_features, frequency_features, Z_Score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset_dir = "E:/increment_classfying/dataset"
feature_dir = 'E:\\increment_classfying\\fea_re_tf' # 保存特征路径
csv_files = [file for file in os.listdir(dataset_dir) if file.endswith('.csv')]
csv_files = sorted(csv_files, key=lambda x: int(x.split('S')[1].split('.')[0]))
hea_files = [file for file in os.listdir(dataset_dir) if file.endswith('.hea')]
hea_files = sorted(hea_files, key=lambda x: int(x.split('S')[1].split('.')[0]))

all_features = []
all_labels = []
#
# feature_names = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11',
#                  'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
#                  'h1', 'h2', 'h3']
feature_names = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11',
                 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',]
count = 0
# 遍历处理每个文件
new_label = 0
for csv_file, hea_file in zip(csv_files, hea_files):
    # 读取.csv文件
    data = pd.read_csv(os.path.join(dataset_dir, csv_file))
    signal = data.iloc[:, 0]  # 假设第一列是信号数据

    # 读取.hea文件获取标签
    with open(os.path.join(dataset_dir, hea_file), 'r') as f:
        label = f.readline().strip()  # 读取第一行作为标签
        if new_label!=label:
            new_label = label
            print('Label:', new_label)


    # 数据预处理和特征提取
    normalized_signal = Z_Score(signal).astype(float)
    t_fea_1 = time_features(signal)
    f_fea_1 = frequency_features(signal, fs=1e8)
    # h_fea = hjorth_cal(signal)
    # t_fea_1 = time_features(normalized_signal)
    # f_fea_1 = frequency_features(normalized_signal, fs=1e8)
    # h_fea = hjorth_cal(normalized_signal)

    # 将特征进行合并
    features = np.concatenate([t_fea_1, f_fea_1])

    # 将特征和标签添加到列表中
    all_features.append(features)
    all_labels.append(label)

    # # 保存特征数组到.npy文件
    # np.save(feature_dir + '\\' + str(csv_file).split('.')[0], features, allow_pickle=True)
    # print(count, "保存成功")
    count += 1

    if int(str(csv_file)[1:].split('.')[0]) >= 4500: ########## 取部分数据test
        break



# 将特征列表转换为 numpy 数组
all_features = np.array(all_features).reshape(4500, 23)
# all_labels = np.array(all_labels)

# 假设 X 是特征集，y 是标签
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

# 使用随机森林作为基础模型
model = RandomForestClassifier()

# 使用递归特征消除进行特征选择
rfe = RFE(model, n_features_to_select=10)  # 选择5个最重要的特征
rfe.fit(X_train, y_train)

# 选择后的特征
selected_features = np.array(feature_names)[rfe.support_]
print("选择的特征:", selected_features.tolist())
# selected_features = np.array(feature_names)[rfe.support_.astype(bool)]
# selected_features = X_train.columns[rfe.support_]

# 使用选择后的特征训练模型并评估性能
X_train_selected = X_train[:, rfe.support_]
X_test_selected = X_test[:, rfe.support_]

model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)

print("模型在测试集上的准确率:", accuracy)


# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
# ticks 坐标轴的坐标点
# label 坐标轴标签说明

indices = range(len(cm))
plt.figure(figsize=(9, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

# plt.xticks(indices, ['normal', 'transient', 'interuption','harmonic', 'meteor', 'flash','cross-modulation','inter-modulation'],rotation=20)
# plt.yticks(indices, ['normal', 'transient', 'interuption','harmonic', 'meteor', 'flash','cross-modulation','inter-modulation'],rotation=0)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

