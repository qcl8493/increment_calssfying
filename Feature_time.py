from scipy.stats import skew
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.stats import kurtosis
from scipy.fft import fft

# conda install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

# 计算时域统计特征
def calculate_time_domain_features(signal):
    mean = np.mean(signal)               #均值
    std = np.std(signal)                 #标准差
    rms = np.sqrt(np.mean(signal**2))    #方差
    skewness = skew(signal)               #斜率
    kurt = kurtosis(signal)               #峰值
    return mean, std, rms, skewness, kurt

# 计算频域统计特征
def calculate_frequency_domain_features(signal, freq_axis):
    signal = signal[:len(freq_axis)]
    mean_freq = np.mean(freq_axis)
    std_freq = np.std(freq_axis)
    rms_freq = np.sqrt(np.mean(freq_axis**2))
    central_freq = np.sum(freq_axis * np.abs(signal)) / np.sum(np.abs(signal))
    kurt_freq = kurtosis(signal)
    return mean_freq, std_freq, rms_freq, central_freq, kurt_freq

dataset_dir = "E:/increment_classfy/dataset"
csv_files = [file for file in os.listdir(dataset_dir) if file.endswith('.csv')]
csv_files = sorted(csv_files, key=lambda x: int(x.split('S')[1].split('.')[0]))
hea_files = [file for file in os.listdir(dataset_dir) if file.endswith('.hea')]
hea_files = sorted(hea_files, key=lambda x: int(x.split('S')[1].split('.')[0]))

all_features = []
all_labels = []
feature_dir = 'E:/increment_classfy/time_features'


# 遍历做TET                                                                           #？？？？？？？？？？？？？？？？
for csv_file, hea_file in zip(csv_files, hea_files):
    # 读取.csv文件
    data = pd.read_csv(os.path.join(dataset_dir, csv_file))
    signal = data.iloc[:, 0] + 1j * data.iloc[:, 1]  # 复数信号  #iloc：通过行、列的索引位置来寻找数据
    print(len(signal))
    # 读取.hea文件
    with open(os.path.join(dataset_dir, hea_file), 'r') as f:
        label = f.readline().strip()  # 读取第一行作为标签
        print(label)

    # # TET变换
    # tfr, Te, GD, TEO, freq_axis = TET_Y(signal.values, hlength=1024, fs=100000000)  # 调整hlength的值以适应您的数据
    # reconstructed_signal = reconstruct_signal(Te)
    # reconstructed_signal = np.real(reconstructed_signal)

    # 计算原始信号的统计特征
    mean, std, rms, skewness, kurt = calculate_time_domain_features(signal)
    # mean_freq, std_freq, rms_freq, central_freq, kurt_freq = calculate_frequency_domain_features(np.abs(fft(signal.values))


    # # 计算瞬态分量的统计特征
    # mean_te, std_te, rms_te, skewness_te, kurt_te = calculate_time_domain_features(reconstructed_signal)
    # print(len(np.abs(fft(reconstructed_signal))))
    # mean_freq_te, std_freq_te, rms_freq_te, central_freq_te, kurt_freq_te = calculate_frequency_domain_features(
    #                                                                      np.abs(fft(reconstructed_signal)), freq_axis)

    # features = np.array([mean, std, rms, skewness, kurt,
    #                      mean_freq, std_freq, rms_freq, central_freq, kurt_freq,
    #                      mean_te, std_te, rms_te, skewness_te, kurt_te,
    #                      mean_freq_te, std_freq_te, rms_freq_te, central_freq_te, kurt_freq_te])

    # features = np.array([mean, std, rms, skewness, kurt,
    #                      mean_freq, std_freq, rms_freq, central_freq, kurt_freq])


    # 保存特征数组到.npy文件
    np.save(feature_dir + '\\' + str(csv_file).split('.')[0], features)    #只取文件名中除后缀之外的内容

    if int(str(csv_file)[1:].split('.')[0]) >= 4500: ##########
        break


for npy_file in sorted(os.listdir(feature_dir)):
    if npy_file.endswith('.npy'):
        features_array = np.load(os.path.join(feature_dir, npy_file)).astype(float)
        all_features.append(features_array)
        # 根据文件名获取标签信息，并添加到标签列表中
        with open(os.path.join(dataset_dir, hea_files[int(npy_file.split('.')[0][1:]) - 1]), 'r') as f:
            label = f.readline().strip()  # 读取第一行作为标签
        all_labels.extend([label] * features_array.shape[0])

# 将列表转换为numpy数组
all_features = np.concatenate(all_features, axis=0)
all_labels = np.array(all_labels)
print(all_labels)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

# 初始化SVM分类器
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

# 在训练集上训练模型
svm_classifier.fit(X_train, y_train)

# 在测试集上进行预测
predictions = svm_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print("准确率:", accuracy)



