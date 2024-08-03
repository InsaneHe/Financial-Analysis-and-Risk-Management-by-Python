# 第二章
# 2020年5月25日投资组合收益率
# 2020年5月25日4只股票的日涨跌幅
return_May25 = [-0.035099, -0.013892, 0.005848, 0.0212421]

# 4只股票的配置权重
weight_list = [0.15, 0.20, 0.25, 0.40]

# 股票数量
n = len(weight_list)

# 创建存放每支股票收益率与配置权重数乘积的空列表
return_weight = []

# 运用for循环
for i in range(n):
    return_weight.append(return_May25[i] * weight_list[i])# 将计算结果存放在列表末尾

# 计算2020年5月25日投资组合的收益率（精确到6位小数）
return_port_May25 = sum(return_weight)
print("2020年5月25日投资组合的收益率", round(return_port_May25, 6))



# NumPy模块
# 导入NumPy
import numpy as np

# 查看NumPy模块版本
print(np.__version__)
# 返回1.26.2



# N维数组（数组是由列表组成的，1个列表（即1个[]）为一维数组（1*n向量），多个列表（即多个[]）为二维数组（m*n矩阵））
# 直接输入法
# 4只股票在投资组合中的配置权重
weight_array1 = np.array([0.15, 0.2, 0.25, 0.4])

# type()查看数据类型
print(type(weight_array1))
# 返回numpy.ndarray

# shape函数查看数组的形状
print(weight_array1.shape)
# 返回(4,)，即为4个元素组成



# 将4只股票的日涨跌幅以数组形式在Python中进行输入
# 输入日站跌幅数据
return_array1 = np.array([[-0.035099, 0.017230, -0.003450, -0.024551, 0.039368],
                          [-0.013892, 0.024334, -0.033758, 0.014622, 0.000128],
                          [0.005848, -0.002907, 0.005831, 0.005797, -0.005764],
                          [0.021242, 0.002133,-0.029803, -0.002743, -0.014301]])

# 查看输出结果
print(return_array1)

# 查看数组的形状
print(return_array1.shape)
# 输出(4, 5)即4行5列的矩阵



# 用array()将列表转换为数组
# 将列表转换为一维数组
weight_array2 = np.array(weight_list)

# 查看结果
print(weight_array2)

# 以列表形式输入日涨跌幅数据
return_list = [-0.035099, 0.017230, -0.003450, -0.024551,
               0.039368, -0.013892, 0.024334, -0.033758, 0.014622, 0.000128,
               0.005848, -0.002907, 0.005831, 0.005797, -0.005764,
               0.021242, 0.002133,-0.029803, -0.002743, -0.014301]

# 转换为一维数组
return_array2 = np.array(return_list)

# 转换为4行5列的二维数组
return_array2 = return_array2.reshape(4, 5)# 使用reshape函数来改变数组的维度

# 查看输出结果
print(return_array2)
# 此处输出的结果与return_array1完全一致

# 用ravel()函数将多维数组降至一维数组
return_array3 = return_array2.ravel()

# 查看输出结果
print(return_array3)
# 输出结果为一维数组



# 查看数组的属性
# ndim函数（查看数组的维度）
print(weight_array1.ndim)
print(return_array1.ndim)

# size函数（查看数组中的元素数量）
print(weight_array1.size)
print(return_array1.size)

# dtype函数（查看数组中的元素类型）
print(weight_array1.dtype)
print(return_array1.dtype)



# 一些特殊的数组
# 整数数列的数组
# arange函数输出一个数组（注意range函数输出的是一个列表）
# 创建0~9的整数数列
a = np.arange(10)

# 查看结果
print(a)

# 创建1~18且步长为3的整数数列
b = np.arange(1, 18, 3)

# 查看结果
print(b)

# 等差数列的数组
# linspace函数快速创建等差数列（参数1：数列的起始值，参数2：数列的终止值，参数3：数列中的元素的个数）
# 创建一个0~100，元素个数为51个的等差数列并以数组格式存放
# 创建等差数列的数组
c = np.linspace(0, 100, 51)

# 查看结果
print(c)

# 查看元素的个数
print(len(c))

# 元素为0的数组
# zeros函数
# 创建一个一维的零数组，数组的元素个数为8
zero_array = np.zeros(8)

# 查看结果
print(zero_array)

# 创建一个二维的零数组（n*m）（参数一：n行，参数二：m列）
# 5行7列
zero_array2 = np.zeros((5, 7))

# 查看结果
print(zero_array2)

# 若已有一个或若干个数组，用zeros_like()函数创建与已有数组相同形状的零数组
# 创建与weight_array1和return_array1相同形状的数组
zero_weight = np.zeros_like(weight_array1)
zero_return = np.zeros_like(return_array1)

# 查看结果
print(zero_weight)
print(zero_return)

# 元素为1的数组（运用ones和ones_like函数，具体输入方式和zeros和zeros_like函数类似）
# 创建与weight_array1和return_array1形状相同且元素均为1的数组
one_weight = np.ones_like(weight_array1)
one_return = np.ones_like(return_array1)

# 查看结果
print(one_weight)
print(one_return)

# 单位矩阵的数组
# 用eye函数创建单位矩阵（即对角线上的元素为1其余元素为0的矩阵）
d = np.eye(6)

# 查看结果
print(d)

# 注：对于NumPy来说，1*n和n*1的数组表现形式都是一样的



# 数组的相关功能
# 索引（投资者可以用于了解自己的投资组合中某只股票在某个交易日的日涨跌幅情况）
# 寻找2020年5月28日中国软件的日涨跌幅（数组中第2行，第4列）
print(return_array1[1, 3])
# 方括号内参数1：第几行，参数2：第几列（都是从0开始）

# 用where函数按一定规则寻找元素在数组中的索引值
# 寻找涨幅超过1.4%的元素的索引值
print(np.where(return_array1 > 0.014))
# 由于此处return_array1是二维数组，元素对应的索引值必然为2个数值，参数1：第几行（即行索引值），参数2：第几列（即列索引值）

# 切片（查找某只股票在若干个交易日或若干只股票在某个交易日或若干只股票在若干个交易日的日涨跌幅情况）
# 提取第2、3行中第2至4列的数据
print(return_array1[1: 3, 1: 4])
# 方括号内，1:3代表选择第2至第3行，1:4代表选择第2列至第4列

# 分别提取第2行的全部数据（中国软件的日涨跌幅），以及第3列的全部数据（2020年5月27日4只股票的日涨跌幅）
# 提取第2行的全部数据
print(return_array1[1])

# 提取第3列的全部数据
print(return_array1[:, 2])

# 排序
# 对股票按照日涨跌幅排序
print(np.sort(return_array1, axis = 0))# 按列对元素由小到大排列，axis=0按列对元素排列，axis=1按行，默认按行排列
print(np.sort(return_array1, axis = 1))# 按行对元素由小到大排列



# 合并（可用append（2个数组）或concatenate函数（2个以上数组））
# 使用append函数合并数组
# 依次以数组形式创建中国银行、上汽集团这两只股票在2020年5月25日至29日期间的日涨跌幅数据然后合并
# 中国银行的日涨跌幅数据
return_BOC = np.array([0.005848, -0.002907, 0.005831, 0.005797, -0.005764])

# 上汽集团的日涨跌幅数据
return_SAIC = np.array([0.021242, 0.002133, -0.029803, -0.002743, -0.014301])

# 按列合并
return_2stock = np.append([return_BOC], [return_SAIC], axis = 0)# axis=0为按列合并，axis=1为按行合并

# 查看结果
print(return_2stock)

# 按行合并
return_2stock_new = np.append([return_BOC], [return_SAIC], axis = 1)# 按行合并

# 查看结果
print(return_2stock_new)

# 用concatenate函数合并数组
# 依次创建中国卫星、中国软件在2020年5月25日至29日期间日涨跌幅数据的两个数组并与中国银行、上汽集团的两个数组结合
# 中国卫星的日涨跌幅数据
return_CAST = np.array([-0.035099, 0.017230, -0.003450, -0.024551, 0.039368])

# 中国软件的日涨跌幅数据
return_CSS = np.array([-0.013892, 0.024334, -0.033758, 0.014622, 0.000128])

# 按列合并
return_4stock = np.concatenate(([return_CAST], [return_CSS], [return_BOC], [return_SAIC]), axis = 0)

# 查看结果
print(return_4stock)

# 按行合并
return_4stock_new = np.concatenate(([return_CAST], [return_CSS], [return_BOC], [return_SAIC]), axis = 1)

# 查看结果
print(return_4stock_new)



# 数组的相关运算
# 数组内的运算
# 求和（sum）
# 依次对return_array1内的每列元素，每行元素和所有元素求和
# 按列求和
return_array1.sum(axis = 0)

# 按行求和
return_array1.sum(axis = 1)

# 全部元素求和
return_array1.sum()



# 求乘积（prod函数）
# 依次对return_array1内的每列元素，每行元素和所有元素求乘积
# 按列求乘积
return_array1.prod(axis = 0)

# 按行求乘积
return_array1.prod(axis = 1)

# 全部元素求乘积
return_array1.prod()



# 求最值（max（最大值），min（最小值））
# 依次对return_array1内的每列元素，每行元素和所有元素求最值
# 按列求最大值
return_array1.max(axis = 0)

# 按行求最大值
return_array1.max(axis = 1)

# 全部元素求最大值
return_array1.max()

# 按列求最小值
return_array1.min(axis = 0)

# 按行求最小值
return_array1.min(axis = 1)

# 全部元素求最小值
return_array1.min()



# 求均值（mean函数）
# 依次对return_array1内的每列元素，每行元素和所有元素求均值
# 按列求均值
return_array1.mean(axis = 0)

# 按行求均值
return_array1.mean(axis = 1)

# 全部元素求均值
return_array1.mean()



# 求方差和标准差（var和std函数）
# 依次对return_array1内的每列元素，每行元素和所有元素求方差和标准差
# 按列求方差
return_array1.var(axis = 0)

# 按行求方差
return_array1.var(axis = 1)

# 全部元素求方差
return_array1.var()

# 按列求标准差
return_array1.std(axis = 0)

# 按行求标准差
return_array1.std(axis = 1)

# 全部元素求标准差
return_array1.std()



# 求开方（sqrt函数），平方（square函数）以及以e为底的指数次方（sqrt函数）
# 依次对return_array1内的每个元素求开方，平方以及以e为底的指数次方
# 对每个元素求开方
np.sqrt(return_array1)
# 开方仅适用于非负数，因此负数的开方在Python中显示为nan，表示无解

# 对每个元素求平方
np.square(return_array1)

# 对每个元素求以e为底的指数次方
np.exp(return_array1)



# 对数运算
# 依次对return_array1内的每个元素计算自然对数（log）、底数为2的对数（log2）、底数为10的对数（log10）和每个元素加1后再求自然对数（loglp）
# 对每个元素计算自然对数
np.log(return_array1)

# 对每个元素计算底数为2的对数
np.log2(return_array1)

# 对每个元素计算底数为10的对数
np.log10(return_array1)

# 对每个元素计算每个元素加1后再求自然对数
np.log1p(return_array1)
# 由于对数仅适用于正数，故负数和0的对数在Python中显示为nan，表示无解



# 数组间的运算
"""
数组间的运算需要遵循以下3个规律：
1，若干个二维数组间的运算，这些数组应具有相同行数和列数（即相同形状）
2，二维数组与一维数组间的运算，一维数组的元素个数应当等于二维数组的列数
3，若干个一维数组间的运算，这些数组应具有相同的元素数量
"""
# 对return_array1和one_return两个数组进行加减乘除和幂运算
# 两个二维数组相加
new_array1 = return_array1 + one_return

# 查看结果
print(new_array1)

# 两个二维数组相减
new_array2 = return_array1 - one_return

# 查看结果
print(new_array2)

# 两个新的二维数组相乘
new_array3 = new_array1 * new_array2

# 查看结果
print(new_array3)

# 两个新的二维数组相除
new_array4 = new_array1 / new_array2

# 查看结果
print(new_array4)

# 两个新的二维数组进行幂运算
# 方法一
new_array5 = new_array1 ** new_array2

# 查看结果
print(new_array5)

# 方法二（pow函数）
new_array6 = pow(new_array1, new_array2)

# 查看结果
print(new_array6)

# 二维数组和一维数组相加
new_array7 = new_array6 + np.array([1, 0, 1, 0, 1])

# 查看结果
print(new_array7)

# 一个数字可以和数组进行运算，但是输出的结果是该数字与数组中每个元素进行运算的结果
# return_array1中每个元素都加1，减1，乘以2，除以2以及进行平方
# 数组的每个元素加1
new_array8 = return_array1 + 1
print(new_array8)

# 数组的每个元素减1
new_array9 = return_array1 - 1
print(new_array9)

# 数组的每个元素乘以2
new_array10 = return_array1 * 2
print(new_array10)

# 数组的每个元素除以2
new_array11 = return_array1 / 2
print(new_array11)

# 数组的每个元素进行平方
new_array12 = return_array1 ** 2
print(new_array12)



# 比较两个或多个形状相同的数组之间对应元素的大小关系，并由此生成包含最大值（maximum）和最小值（minimum）元素的新数组
# 依次创建以return_array1和zero_return之间对应元素的最大值、最小值作为元素的2个新数组
# 创建以两个数组对应元素的最大值作为元素的新数组
return_max = np.maximum(return_array1, zero_return)
print(return_max)

# 创建以两个数组对应元素的最小值作为元素的新数组
return_min = np.minimum(return_array1, zero_return)
print(return_min)



# 矩阵的处理
# 计算矩阵的性质
# 对于return_array1，计算4只股票的日涨跌幅相关系数矩阵（corrcoef）
corr_return = np.corrcoef(return_array1)
print(corr_return)
# 此处相关系数分别为第1行与后面多行分别的相关系数组成新的第一行，第2行与其他各行分别的相关系数组成新的第二行……以此类推

# 查看矩阵的对角线（diag）
np.diag(corr_return)

# 查看矩阵的上三角（triu）
np.triu(corr_return)

# 查看矩阵的下三角（tril）
np.tril(corr_return)

# 查看矩阵的迹（即一个矩阵对角线上个元素的总和）（trace）
np.trace(corr_return)

# 查看矩阵的转置（即将一个矩阵的行和列进行互换）
# 方法一（transpose）
np.transpose(return_array1)

# 方法二（T）
return_array1.T



# 矩阵的运算
# 按照每支股票在投资组合中的配置权重以及每日涨跌幅计算每个交易日投资组合的收益率（即求矩阵的内积）（dot）
# 使用weight_array1和return_array1计算投资组合的日收益率
return_daily = np.dot(weight_array1, return_array1)

# 查看结果
print(return_daily)

# NumPy中的重要子模块linalg用于线性代数运算
# 导入linalg
import numpy.linalg as la

# 使用corr_return为例
# 求矩阵的行列式（det）
la.det(corr_return)

# 求矩阵的逆矩阵（inv）
la.inv(corr_return)
# 可看出原矩阵和逆矩阵的内积是单位矩阵

# 特征值分解（eig）
la.eig(corr_return)
# 输出的结果中，第一个数组为特征值，第二个数组为特征向量（只有m*m矩阵才可以实施特征值分解）

# 奇异值分解（svd）
la.svd(corr_return)
# 输出结果中第1个和第3个数组为酉矩阵，第2个数组为奇异值



# 随机抽样的示例（random模块）
import numpy.random as npr

# 基于正态分布的随机抽样
# 假定从均值为1.5，标准差为2.5的正态分布中抽取随机数，同时设定抽取随机数次数为10万次
# 随机抽样的次数
I = 100000

# 均值
mean1 = 1.5

# 标准差
std1 = 2.5

# 从正态分布中随机抽样
x_norm = npr.normal(loc = mean1, scale = std1, size = I)

# 输出结果
print("从正态分布中随机抽样的均值为", x_norm.mean())
print("从正态分布中抽样的标准差", x_norm.std())

# 由于是随机抽样，故不同组抽样得到的结果会有所不同，但不会很大



# 从标准正态分布中抽取随机数，抽取随机数的次数仍然为10万次（randn，standard_normal，normal三个函数供选择）
# 运用randn函数
x_snorm1 = npr.randn(I)

# 运用standard_normal函数
x_snorm2 = npr.standard_normal(size = I)

# 均值
mean2 = 0

# 标准差
std2 = 1

# 运用normal函数
x_snorm3 = npr.normal(loc = mean2, scale = std2, size = I)

# 输出结果
print("运用randn函数从标准正态分布中抽样的均值", x_snorm1.mean())
print("运用randn函数从标准正态分布中抽样的标准差", x_snorm1.std())
print("运用standard_normal函数从标准正态分布中抽样的均值", x_snorm2.mean())
print("运用standard_normal函数从标准正态分布中抽样的标准差", x_snorm2.std())
print("运用normal函数从标准正态分布中抽样的均值", x_snorm3.mean())
print("运用normal函数从标准正态分布中抽样的标准差", x_snorm3.std())

# 由上可得：用不同的函数从标准正态分布中抽取随机数所得结果比较类似



# 基于对数正态分布的随机抽样
# 假定随机变量x的自然对数服从均值为0.4、标准差为1.2的正态分布，对变量x进行随机抽样，抽取随机数的次数依然为10万次
# 均值
mean3 = 0.4

# 标准差
std3 = 1.2

# 从对数正态分布中随机抽样
x_logn = npr.lognormal(mean = mean3, sigma = std3, size = I)

# 输出结果
print("从对数正态分布中抽样的均值", x_logn.mean())
print("从对数正态分布中抽样的标准差", x_logn.std())



# 基于卡方分布的随机抽样
# 假定从自由度为6和98的卡方分布中抽取随机数，并且抽取随机数的次数依然是10万次
# 设置自由度
freedom1 = 6
freedom2 = 98

# 从自由度为6的卡方分布中随机抽样
x_chi1 = npr.chisquare(df = freedom1, size = I)

# 从自由度为98的卡方分布中随机抽样
x_chi2 = npr.chisquare(df = freedom2, size = I)

# 输出结果
print("从自由度为6的卡方分布中抽样的均值", x_chi1.mean())
print("从自由度为6的卡方分布中抽样的标准差", x_chi1.std())
print("从自由度为6的卡方分布中抽样的均值", x_chi2.mean())
print("从自由度为6的卡方分布中抽样的标准差", x_chi2.std())

# 由上可得：随着自由度增大，抽取随机数的均值和标准差都会增大



# 基于学生t分布的随机抽样
# 假定从自由度为3和130的学生t分布中抽取随机数，抽取随机数的次数依然是10万次
# 设置自由度
freedom3 = 3
freedom4 = 130

# 从自由度为3的学生t分布中随机抽样
x_t1 = npr.standard_t(df = freedom3, size = I)
x_t2 = npr.standard_t(df = freedom4, size = I)

# 输出结果
print("从自由度为3的学生t分布中抽样的均值", x_t1.mean())
print("从自由度为3的学生t分布中抽样的标准差", x_t1.std())
print("从自由度为3的学生t分布中抽样的均值", x_t2.mean())
print("从自由度为3的学生t分布中抽样的标准差", x_t2.std())

# 由上可得：随着自由度不断提升，学生t分布不断接近于标准正态分布



# 基于F分布的随机抽样
# 假定从自由度为n1 = 4和n2 = 10的F分布中抽取随机数，抽取随机数的次数依然为10万次
# 设置自由度
freedom5 = 4
freedom6 = 10

# 从F分布中抽取随机数
x_f = npr.f(dfnum = freedom5, dfden = freedom6, size = I)

# 输出结果
print("从F分布中抽样的均值", x_f.mean())
print("从F分布中抽样的标准差", x_f.std())



# 基于贝塔分布的随机抽样
# 假定从α = 3，β = 7的贝塔分布中抽取随机数，抽取随机数的次数依然为10万次
# 贝塔分布的第1个参数
a1 = 3

# 贝塔分布的第2个参数
b1 = 7

# 从贝塔分布中抽取随机数
x_beta = npr.beta(a = a1, b = b1, size = I)

# 输出结果
print("从贝塔分布中抽样的均值", x_beta.mean())
print("从贝塔分布中抽样的标准差", x_beta.std())



# 基于伽马分布的随机抽样
# 假定从α = 2，β = 8的伽马分布中抽取随机数，抽取随机数的次数依然是10万次
# 形状参数
a2 = 2

# 尺度参数
b2 = 8

# 从伽马分布中抽取随机数
x_gamma = npr.gamma(shape = a2, scale = b2, size = I)

# 输出结果
print("从伽马分布中抽样的均值", x_gamma.mean())
print("从伽马分布中抽样的标准差", x_gamma.std())



# 现金流模型
# 使用numpy-financial模块
import numpy_financial as npf

# 查看版本信息
print(npf.__version__)

# 现金流终值
# 计算项目终值时使用fv函数
# 计算P65页例2-40的项目终值
# 初始投资金额
V0 = 2e7

# 每年固定金额投资
V1 = 3e6

# 投资期限（年）
T = 5

# 年化投资回报率
r = 0.08

# 计算项目终值并且期间追加投资发生在每年年末
FV1 = npf.fv(rate = r, nper = T, pmt = -V1, pv = -V0, when = "end")

# 输出结果
print("计算得到项目终值（期间追加投资发生在每年年末）", round(FV1, 2))

# 计算项目终值并且期间追加投资发生在每年年初
FV2 = npf.fv(rate = r, nper = T, pmt = -V1, pv = -V0, when = "begin")

# 输出结果
print("计算得到项目终值（期间追加投资发生在每年年初）", round(FV2, 2))

# 期间追加投资发生时点不同而导致项目终值的差异
FV_diff = FV2 - FV1

# 输出结果
print("期间追加投资发生时点不同而导致项目终值的差异", round(FV_diff, 2))

# 现金流现值（pv函数）
# 针对例2-41并运用函数fv计算项目现值
# 期间每年产生的现金流入
V1 = 2e6

# 期末一次性现金流入
Vt = 2.5e7

# 投资期限（年）
T = 6

# 投资回报率
R = 0.06

# 计算项目现值并且期间现金流发生在每年年末
PV1 = npf.pv(rate = R, nper = T, pmt = V1, fv = Vt, when = 0)

# 输出结果
print("计算得到项目现值（期间现金流发生在每年年末）", round(PV1, 2))

# 计算项目现值并且期间现金流发生在每年年初
PV2 = npf.pv(rate = R, nper = T, pmt = V1, fv = Vt, when = 1)

# 输出结果
print("计算得到项目现值（期间现金流发生在每年年初）", round(PV2, 2))

# 期间现金流发生时点不同而导致项目现值的差异
PV_diff = PV2 - PV1

# 输出结果
print("期间现金流发生时点不同而导致项目现值的差异", round(PV_diff, 2))



# 净现值与内含报酬率
# 净现值（npv函数）
# 针对例2-42并运用函数npv计算项目的净现值
import numpy as np

# 投资回报率
R1 = 0.09

# 项目的净现金流（数组格式）
cashflow = np.array([-2.8e7, 7e6, 8e6, 9e6, 1e7])

# 计算项目的净现值
NPV1 = npf.npv(rate = R1, values = cashflow)

# 输出结果
print("计算得到项目净现值", round(NPV1, 2))

# 新的投资回报率
R2 = 0.06

# 计算项目新的净现值
NPV2 = npf.npv(rate = R2, values = cashflow)

# 输出结果
print("计算得到项目新的净现值", round(NPV2, 2))

# 由上可得：当投资回报率为9%时，该项目净现值为负数，当投资回报率为6%时，项目净现值为正数



# 内含报酬率（即投资回报率为多少时，项目的净现值为0，当项目的内含报酬率高于预期收益率时，该项目可行）
# 使用numpy_financial模块的函数irr
# 沿用例2-42中的项目信息和数据，计算该项目的内含报酬率
# 计算项目的内含报酬率
IRR = npf.irr(values = cashflow)

# 保留小数点后6位
print("计算得到项目的内含报酬率", round(IRR, 6))



# 住房按揭贷款的等额本息还款
# 使用Python计算等额本息还款金额时使用numpy_financial模块的函数pmt/impt函数（计算等额本息还款利息部分）/ppmt函数（计算等额本息还款本金部分）
# impt和ppmt都需要输入参数per（表示逐次还款的期限并且用数组表示）
# 计算例2-44中的每月还款总金额、还款利息和还款本金
# 住房按揭贷款本金
prin_loan = 5e6

# 贷款期限（月）
tenor_loan = 5 * 12

# 贷款月利率
rate_loan = 0.06 / 12

# 计算住房按揭贷款每月还款总金额
payment = npf.pmt(rate = rate_loan, nper = tenor_loan, pv = prin_loan, fv = 0, when = "end")
# rate：住房按揭贷款月利率，nper：贷款的整体期限（月），pv：住房按揭贷款的本金金额，fv：期末的现金流（通常为0），默认值为0，
# when：每月还款的发生时点，期初为1/"begin"，期末为0/"end"

# 输出结果
print("计算得到住房按揭贷款每月还款总金额", round(payment, 2))

# 创建包含每次还款期限的数组
tenor_list = np.arange(tenor_loan) + 1
# +1是因为np.arange用于创造等差数列，从0开始，故+1来使数列从1开始

# 计算住房按揭贷款每月偿还的利息金额
payment_interest = npf.ipmt(rate = rate_loan, per = tenor_list, nper = tenor_loan, pv = prin_loan, fv = 0, when = "end")

# 输出结果
print(payment_interest)

# 计算住房按揭贷款每月偿还的本金金额
payment_principle = npf.ppmt(rate = rate_loan, per = tenor_list, nper = tenor_loan, pv = prin_loan, fv = 0, when = "end")

# 输出结果
print(payment_principle)

# 验证是否与每月还款总金额保持一致
print((payment_principle + payment_interest).round(2))

# 由上可得：该住房按揭贷款的每月还款总金额为96664.01元，且每月偿还的利息金额逐月递减，每月偿还的本金逐月递增
