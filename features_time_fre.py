import numpy as np
import scipy.stats
import pandas as pd
import os
import matplotlib.pyplot as plt

# 时域统计特征
def time_features(signal):
    # mean = np.mean(signal)
    # std = np.nanstd(signal)
    # rms = np.sqrt(np.mean(signal**2))
    # skewness = skew(signal)
    # kurt = kurtosis(signal)
    # return mean, std, rms, skewness, kurt
    N = len(signal)
    y = signal
    t_mean_1 = np.mean(y)  # 1_均值（平均幅值）

    t_std_2 = np.std(y, ddof=1)  # 2_标准差

    t_fgf_3 = ((np.mean(np.sqrt(np.abs(y))))) ** 2  # 3_方根幅值

    t_rms_4 = np.sqrt((np.mean(y ** 2)))  # 4_RMS均方根

    t_pp_5 = 0.5 * (np.max(y) - np.min(y))  # 5_峰峰值  (参考周宏锑师姐 博士毕业论文)

    # t_skew_6   = np.sum((t_mean_1)**3)/((N-1)*(t_std_3)**3)
    t_skew_6 = scipy.stats.skew(y)  # 6_偏度 skewness

    # t_kur_7   = np.sum((y-t_mean_1)**4)/((N-1)*(t_std_3)**4)
    t_kur_7 = scipy.stats.kurtosis(y)  # 7_峭度 Kurtosis

    t_cres_8 = np.max(np.abs(y)) / t_rms_4  # 8_峰值因子 Crest Factor

    t_clear_9 = np.max(np.abs(y)) / t_fgf_3  # 9_裕度因子  Clearance Factor

    t_shape_10 = (N * t_rms_4) / (np.sum(np.abs(y)))  # 10_波形因子 Shape fator

    t_imp_11 = (np.max(np.abs(y))) / (np.mean(np.abs(y)))  # 11_脉冲指数 Impulse Fator

    t_fea = np.array([t_mean_1, t_std_2, t_fgf_3, t_rms_4, t_pp_5,
                      t_skew_6, t_kur_7, t_cres_8, t_clear_9, t_shape_10, t_imp_11])
    # t_fea = np.array([t_mean_1, t_std_2, t_rms_4,
    #                   t_skew_6, t_kur_7, t_cres_8, t_shape_10, t_imp_11])

    # print("t_fea:",t_fea.shape,'\n', t_fea)
    return t_fea


# 频域统计特征
def frequency_features(signal, fs):
    L = len(signal)
    PL = abs(np.fft.fft(signal / L))[:int(L / 2)]
    PL[0] = 0
    f = np.fft.fftfreq(L, 1 / fs)[:int(L / 2)]
    x = f
    y = PL
    K = len(y)

    # fft_sig = np.fft.fft(signal)
    # freq_bins = np.fft.fftfreq(len(fft_sig)) * fs

    # mean_freq = np.mean(y)
    # weighted_average_fre = (np.sum(x * y)) / (np.sum(y))  # 频谱的加权平均频率（频率的期望值）
    #
    # std_freq = np.std(np.abs(fft_sig))
    # rms_freq = np.sqrt(np.mean(np.abs(fft_sig**2)))
    # # central_freq = np.sum(np.abs(fft_sig) * freq_bins) / np.sum(np.abs(fft_sig))
    # amplitude = np.abs(fft_sig) * np.abs(fft_sig) * np.abs(fft_sig)  # 使用幅度的平方作为权重
    # central_freq = np.sum(freq_bins * amplitude) / np.sum(amplitude)
    # # kurt_freq = kurtosis(np.abs(fft_sig))
    # kurt_freq = (np.sum((y - mean_amplitude) ** 4)) / (K * ((variance_amplitude) ** 2))
    f_12 = np.mean(y)

    f_13 = np.var(y)

    f_14 = (np.sum((y - f_12) ** 3)) / (K * ((np.sqrt(f_13)) ** 3))

    f_15 = (np.sum((y - f_12) ** 4)) / (K * ((f_13) ** 2))

    f_16 = (np.sum(x * y)) / (np.sum(y))

    f_17 = np.sqrt((np.mean(((x - f_16) ** 2) * (y))))

    f_18 = np.sqrt((np.sum((x ** 2) * y)) / (np.sum(y)))

    f_19 = np.sqrt((np.sum((x ** 4) * y)) / (np.sum((x ** 2) * y)))

    f_20 = (np.sum((x ** 2) * y)) / (np.sqrt((np.sum(y)) * (np.sum((x ** 4) * y))))

    f_21 = f_17 / f_16

    f_22 = (np.sum(((x - f_16) ** 3) * y)) / (K * (f_17 ** 3))

    f_23 = (np.sum(((x - f_16) ** 4) * y)) / (K * (f_17 ** 4))

    f_fea = np.array([f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23])

    return f_fea

def Z_Score(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    normal_sig = (signal - mean) / std
    return normal_sig

def hjorth_cal(x, axis=-1):
    # 输入1维或N维数据

    # 返回activity、Mobility、complexity

    x = np.asarray(x)
    # Calculate derivatives
    dx = np.diff(x, axis=axis)           #一级差分
    ddx = np.diff(dx, axis=axis)           #二级差分
    # 方差计算
    x_var = np.var(x, axis=axis)
    dx_var = np.var(dx, axis=axis)
    ddx_var = np.var(ddx, axis=axis)


    act = x_var
    # Mobility and complexity
    mob = np.sqrt(dx_var / x_var)
    com = np.sqrt(ddx_var / dx_var) / mob
    hjorth = np.array([act, mob, com])
    return hjorth