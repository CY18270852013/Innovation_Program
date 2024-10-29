import numpy as np

# 生成正态分布样本
samples = np.random.normal(loc=5, scale=5, size=1000)

# 计算最大值、最小值、均值、标准差、四分位数
max_val = np.max(samples)
min_val = np.min(samples)
mean_val = np.mean(samples)
std_val = np.std(samples)
quartile_1 = np.percentile(samples, 25)
quartile_2 = np.percentile(samples, 50)
quartile_3 = np.percentile(samples, 75)

# 输出结果
print("max value: {:.3f}".format(max_val))
print("min value: {:.3f}".format(min_val))
print("mean value: {:.3f}".format(mean_val))
print("standard deviation: {:.3f}".format(std_val))
print("1st quartile: {:.3f}".format(quartile_1))
print("2st quartile: {:.3f}".format(quartile_2))
print("3st quartile: {:.3f}".format(quartile_3))

# 打印数组数据类型和内存大小
print("Array data type: {}".format(samples.dtype))
print("Array size: {} bytes".format(samples.nbytes))

# 转换数据类型并重新计算内存大小
samples = samples.astype(np.int32)
print("New array size: {} bytes".format(samples.nbytes))