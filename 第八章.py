# 第八章
# 股票市场简介
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["FangSong"]
mpl.rcParams["axes.unicode_minus"] = False
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# 从外部导入数据
index_data = pd.read_excel("C:/Users/InsaneHe/desktop/Python/四只A股市场股指的日收盘价数据（2018-2020）.xlsx", sheet_name = "Sheet1",
                           header = 0, index_col = 0)

# 数据可视化
index_data.plot(subplots = True, layout = (2, 2), figsize = (10, 10), fontsize = 13, grid = True)
plt.subplot(2, 2, 1)# 第1张子图
plt.ylabel(u"指数点位", fontsize = 11, position = (0, 0))# 增加第1张子图的纵坐标标签
# 可看出4只股指在走势形态上是趋同的且有明显的深市强，沪市弱的局面



# 股票内在价值
# 零增长模型
# 通过Python自定义一个运用零增长模型计算股票内在价值的函数
def value_ZGM(D, r):
    """
    定义一个运用零增长模型计算股票内在价值的函数
    D：代表企业已支付的最近一期每股股息金额
    r：代表与企业的风险相匹配的贴现利率（每年复利1次）
    """
    value = D / r# 计算股票的内在价值
    return value

# 2020年12月31日，使用零增长模型计算招商银行A股的内在价值，最近一期支付的股息为2020年7月10日的1.2元/股，未来每年支付股息均为1.2元/股，贴现利率为11.18%
# 招商银行A股的固定股息
Div = 1.2

# 贴现利率
rate = 0.1118

# 计算股票内在价值
value = value_ZGM(D = Div, r = rate)
print("运用零增长模型计算招商银行A股股票内在价值", round(value, 4))



# 不变增长模型
# 通过Python自定义一个运用不变增长模型计算股票内在价值的函数
def value_CGM(D, g, r):
    """
    定义一个运用不变增长模型计算股票内在价值的函数
    D：代表企业已支付的最近一期每股股息金额
    g：代表企业的股息增长率，并且数值要小于贴现利率
    r：代表与企业的风险相匹配的贴现利率（每年复利1次）
    """
    if r > g:# 当贴现利率大于股息增长率
        value = D * (1 + g) / (r - g)
    else:# 当贴现利率小于或等于股息增长率
        value = "输入的贴现利率小于或等于股息增长率而导致结果不存在"
    return value

# 沿用上例信息，运用不变增长模型重新计算招商银行A股股票的内在价值，在估计股息增长率时，依据最近5年（2016-2020）招商银行股息支付情况结合对该银行未来经营情况的预测将每年股息增长率确定为10%
# 招商银行股息增长率
growth = 0.1

# 计算股票内在价值
value_new = value_CGM(D = Div, g = growth, r = rate)
print("运用不变增长模型计算招商银行A股股票的内在价值", round(value_new, 4))



# 二阶段增长模型
# 通过Python自定义一个运用二阶段增长模型计算股票内在价值的函数
def value_2SGM(D, g1, g2, T, r):
    """
    定义一个运用二阶段增长模型计算股票内在价值的函数
    D：代表企业已支付的最近一期每股股息金额
    g1：代表企业在第1个阶段的股息增长率
    g2：代表企业在第2个阶段的股息增长率，并且数值要小于贴现利率
    T：代表企业第1个阶段的期限，单位是年
    r：代表与企业的风险相匹配的提贴现利率（每年复利1次）
    """
    if r > g2:# 贴现利率大于第2个阶段的股息增长率
        T_list = np.arange(1, T + 1)# 创建从1到T的整数数列
        V1 = D * np.sum(pow(1 + g1, T_list) / pow(1 + r, T_list))# 计算第1个阶段股息贴现之和
        V2 = D * pow(1 + g1, T) * (1 + g2) / (pow(1 + r, T) * (r - g2))# 计算第2个阶段股息贴现之和
        value = V1 + V2# 计算股票的内在价值
    else:# 贴现利率小于或等于第2阶段的股息增长率
        value = "输入的贴现利率小于或等于第2阶段的股息增长率而导致结果不存在"
    return value

# 沿用上例信息，运用二阶段增长模型重新计算招商银行A股股票的内在价值，依据过去招商银行股息第1阶段股息增长率为11%，第2阶段股息增长率为8%，第1阶段10年，其他变量取值与之前一致
# 第1阶段的股息增长率
g_stage1 = 0.11

# 第2阶段的股息增长率
g_stage2 = 0.08

# 第1阶段的期限（年）
T_stage1 = 10

# 计算股票内在价值
value_2stages = value_2SGM(D = Div, g1 = g_stage1, g2 = g_stage2, T = T_stage1, r = rate)
print("运用二阶段增长模型计算招商银行A股股票内在价值", round(value_2stages, 4))



# 敏感性分析
# 分析不同阶段的股息增长率与股票内在价值之间的关系。同时，第1阶段股息增长率取值为[6%, 11%]区间的等差数列，第2阶段股息增长率取值为[3%, 8%]区间的等差数列
# 第1个阶段股息增长率的数组
g1_list = np.linspace(0.06, 0.11, 100)

# 第2个阶段股息增长率的数组
g2_list = np.linspace(0.03, 0.08, 100)

# 创建存放对应第1个阶段股息增长率变化的股票内在价值初始数组
value_list1 = np.zeros_like(g1_list)

# 运用for语句
for i in range(len(g1_list)):
    value_list1[i] = value_2SGM(D = Div, g1 = g1_list[i], g2 = g_stage2, T = T_stage1, r = rate)# 计算股票内在价值

# 创建存放对应第2个阶段股息增长率变化的股票内在价值初始数组
value_list2 = np.zeros_like(g2_list)

# 运用for语句
for i in range(len(g2_list)):
    value_list2[i] = value_2SGM(D = Div, g1 = g_stage1, g2 = g2_list[i], T = T_stage1, r = rate)# 计算股票内在价值

plt.figure(figsize = (11, 6))
plt.subplot(1, 2, 1)# 第1行第1列的子图
plt.plot(g1_list, value_list1, "r-", lw = 2.5)
plt.xticks(fontsize = 13)
plt.xlabel(u"第1个阶段股息增长率", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylabel(u"股票内在价值", fontsize = 13)
plt.title(u"第1个阶段股息增长率与股票内在价值的关系图", fontsize = 14)
plt.grid()
plt.subplot(1, 2, 2, sharey = plt.subplot(1, 2, 1))# 第1行第2列子图
plt.plot(g2_list, value_list2, "b-", lw = 2.5)
plt.xticks(fontsize = 13)
plt.xlabel(u"第2个阶段股息增长率", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"第2个阶段股息增长率与股票内在价值的关系图", fontsize = 14)
plt.grid()
plt.show()
# 可看出：在二阶段增长模型中无论第1阶段股息增长率还是第2阶段股息增长率，都对股票内在价值产生了正效应，且股票内在价值对第2阶段股息增长率更敏感



# 三阶段增长模型
# 通过Python自定义一个运用三阶段增长模型计算股票内在价值的函数
def value_3SGM(D, ga, gb, Ta, Tb, r):
    """
    定义一个运用三阶段增长模型计算股票内在价值的函数
    D：代表企业已支付的最近一期每股股息金额
    ga：代表企业在第1阶段的股息增长率
    gb：代表企业在第3阶段的股息增长率，并且数值要小于贴现利率
    Ta：代表企业第1阶段的期限（年）
    Tb：代表企业第1阶段和第2阶段的期限之和（年）
    r：代表与企业的风险相匹配的贴现利率（每年复利1次）
    """
    # 为更好地理解代码的编写逻辑，分为以下4个步骤
    # 第1步：计算第1阶段股息贴现之和
    if r > gb:# 贴现利率大于第3阶段的股息增长率
        Ta_list = np.arange(1, Ta + 1)# 创建从1到Ta的自然数数组
        D_stage1 = D * pow((1 + ga), Ta_list)# 计算第1阶段每期股息金额的数组
        V1 = np.sum(D_stage1 / pow(1 + r, Ta_list))# 计算第1阶段股息贴现之和
    # 第2步：计算第2阶段股息贴现之和
        Tb_list = np.arange(Ta + 1, Tb + 1)# 创建从Ta+1到Tb的自然数数组
        D_t = D_stage1[-1]# 第1阶段最后一期股息
        D_stage2 = []# 创建存放第2阶段的每期股息金额
        for i in range(len(Tb_list)):
            gt = ga - (ga - gb) * (Tb_list[i] - Ta) / (Tb - Ta)# 依次计算第2阶段的每期股息增长率
            D_t = D_t * (1 + gt)# 依次计算第2阶段的每期股息金额
            D_stage2.append(D_t)# 将计算得到的每期股息添加至列表尾部
        D_stage2 = np.array(D_stage2)# 将列表转换为数组格式
        V2 = np.sum(D_stage2 / pow(1 + r, Tb_list))# 计算第2阶段股息贴现之和
    # 第3步：计算第3阶段股息贴现之和
        D_Tb = D_stage2[-1]# 第2阶段最后一期股息
        V3 = D_Tb * (1 + gb) / (pow(1+r, Tb) * (r - gb))# 计算第3阶段股息贴现之和
    # 第4步：计算股票内在价值
        value = V1 + V2 + V3# 计算股票内在价值
    else:# 贴现利率小于或等于第3阶段的股息增长率
        value = "输入的贴现利率小于或等于第3阶段的股息增长率而导致结果不存在"
    return value

# 沿用上例信息，运用三阶段增长模型重新计算招商银行A股股票的内在价值，
# 依据过去信息，设招商银行股息第1阶段股息增长率为11%，持续6年，第2阶段股息增长率以线性方式从11%下降到7.5%，持续4年，第3阶段股息增长率为7.5%，其他变量取值与之前一致
# 第1阶段的股息增长率
g_stage1 = 0.11

# 第3阶段的股息增长率
g_stage3 = 0.075

# 第1阶段的年限
T_stage1 = 6

# 第2阶段的年限
T_stage2 = 4

# 计算股票内在价值
value_3stages = value_3SGM(D = Div, ga = g_stage1, gb = g_stage3, Ta = T_stage1, Tb = T_stage1 + T_stage2, r = rate)
print("运用三阶段增长模型计算招商银行A股股票内在价值", round(value_3stages, 4))



# 借助Python并运用敏感性分析考察最近一期已支付的股息金额、贴现利率、第1阶段股息增长率和第3阶段股息增长率对股票内在价值的影响并进行可视化
# 最近一期已支付的股息金额的取值是处于[0.8, 1.6]区间的等差数列，贴现利率的取值是处于[8%, 12%]区间的等差数列，
# 第1阶段股息增长率取值是处于[7%, 11%]区间的等差数列，第3阶段股息增长率的取值是处于[4%, 8%]区间的等差数列
# 第1步：计算对应每个变量不同取值的股票内在价值
# 最近一期已支付的股息金额的数组
Div_list = np.linspace(0.8, 1.6, 100)

# 贴现利率的数组
rate_list = np.linspace(0.08, 0.12, 100)

# 第1阶段股息增长率的数组
ga_list = np.linspace(0.07, 0.11, 100)

# 第3阶段股息增长率的数组
gb_list = np.linspace(0.04, 0.08, 100)

# 创建对应不同股息金额的股票内在价值初始数组
value_list1 = np.zeros_like(Div_list)

# 运用for语句
for i in range(len(Div_list)):
    value_list1[i] = value_3SGM(D = Div_list[i], ga = g_stage1, gb = g_stage3, Ta = T_stage1, Tb = T_stage1 + T_stage2,
                                r = rate)# 计算股票内在价值

# 创建对应不同贴现利率的股票内在价值初始数据
value_list2 = np.zeros_like(rate_list)

# 运用for语句
for i in range(len(rate_list)):
    value_list2[i] = value_3SGM(D = Div, ga = g_stage1, gb = g_stage3, Ta = T_stage1, Tb = T_stage1 + T_stage2, r = rate_list[i])

# 创建对应第1阶段不同股息增长率的股票内在价值初始数组
value_list3 = np.zeros_like(ga_list)

# 运用for语句
for i in range(len(ga_list)):
    value_list3[i] = value_3SGM(D = Div, ga = ga_list[i], gb = g_stage3, Ta = T_stage1, Tb = T_stage1 + T_stage2, r = rate)

# 创建对应第3阶段不同股息增长率的股票内在价值初始数组
value_list4 = np.zeros_like(gb_list)

# 运用for语句
for i in range(len(gb_list)):
    value_list4[i] = value_3SGM(D = Div, ga = g_stage1, gb = gb_list[i], Ta = T_stage1, Tb = T_stage1 + T_stage2, r = rate)

# 第2步：将以上结果可视化并且以2*2的子图形式展示
plt.figure(figsize = (10, 11))
plt.subplot(2, 2, 1)# 第1行第1列的子图
plt.plot(Div_list, value_list1, "r-", lw = 2.5)
plt.xticks(fontsize = 13)
plt.xlabel(u"最近一期已支付的股息金额", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylabel(u"股票内在价值", fontsize = 13, rotation = 90)
plt.grid()
plt.subplot(2, 2, 2)# 第1行第2列的子图
plt.plot(rate_list, value_list2, "b-", lw = 2.5)
plt.xticks(fontsize = 13)
plt.xlabel(u"贴现利率", fontsize = 13)
plt.yticks(fontsize = 13)
plt.grid()
plt.subplot(2, 2, 3)# 第2行第1列的子图
plt.plot(ga_list, value_list3, "r-", lw = 2.5)
plt.xticks(fontsize = 13)
plt.xlabel(u"第1个阶段股息增长率", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylabel(u"股票内在价值", fontsize = 13, rotation = 90)
plt.grid()
plt.subplot(2, 2, 4)# 第2行第2列的子图
plt.plot(gb_list, value_list4, "b-", lw = 2.5)
plt.xticks(fontsize = 13)
plt.xlabel(u"第3个阶段股息增长率", fontsize = 13)
plt.yticks(fontsize = 13)
plt.grid()
plt.show()
# 可看出：在三阶段增长模型中，最近一期已支付的股息金额、第1阶段股息增长率这两个变量与股票内在价值之间呈现正向的线性关系；
# 第3阶段股息增长率与股票内在价值之间呈现正向的非线性关系；
# 贴现利率与股票内在价值之间则呈现反向的非线性关系



# 股票价格服从的随机过程
# 运用招商银行A股2018年至2020年的日收盘价数据，模拟未来3年（2021年至2023年）该股票的每日价格走势，模拟路径共500条。
# 同时在模拟过程中将初始值设定为2021年1月4日（2021年首个交易日）的收盘价43.17元/股
# 第1步：导入数据并且计算得到招商银行A股的平均年化收益率和年化波动率
# 导入外部数据
S = pd.read_excel("C:/Users/InsaneHe/desktop/Python/招商银行A股日收盘价数据（2018-2020年）.xlsx", sheet_name = "Sheet1",
                  header = 0, index_col = 0)

# 计算招商银行A股日收益率。利用8.4.1节的式（8-45）计算
R = np.log(S / S.shift(1))

# 股票的平均年化收益率
mu = R.mean() * 252

# 转换为浮点型数据类型
mu = float(mu)
print("招商银行A股平均年化收益率", round(mu, 6))

# 股票收益的年化波动率
sigma = R.std() * np.sqrt(252)
sigma = float(sigma)
print("招商银行A股年化波动率", round(sigma, 6))

# 第2步：输入需要进行模拟的相关参数，并运用3.1.4节讨论的函数date_range，通过该函数创建从2021年1月4日至2023年12月末并且是工作日的时间数列
# 导入NumPy的子模块random
import numpy.random as npr

# 创建2021年至2023年的工作日数列
date = pd.date_range(start = "2021-01-04", end = "2023-12-31", freq = "B")

# 计算date的元素个数
N = len(date)

# 设定模拟的路径数量（随机抽样的次数）
I = 500

# 单位时间的长度（1天）
dt = 1.0 / 252

# 创建存放模拟服从几何布朗运动的未来股价初始数组
S_GBM = np.zeros((N, I))

# 模拟的起点设为2021年1月4日的收盘价
S_GBM[0] = 43.17

# 第3步：运用for语句创建模拟的未来股价时间序列
for t in range(1, N):
    epsilon = npr.standard_normal(I)# 基于标准正态分布的随机抽样
    S_GBM[t] = S_GBM[t-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * epsilon * np.sqrt(dt))# 计算未来每个工作日的股价

# 转换为带有时间索引的数据框
S_GBM = pd.DataFrame(S_GBM, index = date)

# 显示数据框的开头5行
print(S_GBM.head())

# 显示数据框的末尾5行
print(S_GBM.tail())

# 显示数据框的描述性统计指标
print(S_GBM.describe())

# 第4步：将招商银行A股模拟股价的结果进行可视化
plt.figure(figsize = (9, 6))
plt.plot(S_GBM)
plt.xlabel(u"日期", fontsize = 13)
plt.ylabel(u"招商银行股价", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"2021-2023年服从几何布朗运动的股价模拟路径", fontsize = 13)
plt.grid()
plt.show()
# 可看出：未来3年的股价绝大多数为20元/股到150元/股

# 将本次模拟的前20条路径进行可视化
plt.figure(figsize = (9, 6))
plt.plot(S_GBM.iloc[: , 0: 20])
plt.xlabel(u"日期", fontsize = 13)
plt.ylabel(u"招商银行股价", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"2021-2023年服从几何布朗运动的股价的前20条模拟路径", fontsize = 13)
plt.grid()
plt.show()
# 可看出：在模拟的20条路径中，第3年末（2023年年末）的股价最低接近20元/股，最高则接近120元/股



# 构建股票最优投资组合
# 投资组合的主要变量
# 投资组合的预期收益率
# 创建投资组合中每支股票的随机权重并确保权重合计数等于1（rand函数）
# 从均匀分布中随机抽取5个随机数
x = npr.rand(5)

# 创建权重数组
weight = x / np.sum(x)

# 输出结果
print(weight)

# 验证权重随机数之和是否等于1
print(round(sum(weight), 2))



# 假设C金融机构管理的股票投资组合配置了5只A股股票，具体为长江电力、平安银行、上海机场、中信证券以及顺丰控股，
# 选取的股价数据是2018年至2020年期间每个交易日的收盘价，假定每支股票配置的权重均为20%
# 第1步：导入股票的收盘价数据并且进行可视化
# 从外部导入数据
data_stocks = pd.read_excel("C:/Users/InsaneHe/desktop/Python/5只A股股票的收盘价（2018年至2020年）.xlsx",
                            sheet_name = "Sheet1", header = 0, index_col = 0)

# 将股价按照首个交易日进行归一化处理并且可视化
(data_stocks / data_stocks.iloc[0]).plot(figsize = (9, 6), grid = True)
# 可看出：投资组合配置的5只股票在2018年至2020年期间都收获了一定的涨幅，产生了财富效应

# 第2步：按照公式构建这5只股票日收益率的时间序列，同时进行可视化
# 计算股票的对数收益率
R = np.log(data_stocks / data_stocks.shift(1))

# 输出描述性统计指标
R.describe()

# 将股票收益率用直方图显示
R.hist(bins = 40, figsize = (9, 11))
# 目测得：5只股票的日收益率均不满足正态分布。同时相比其他3只股票，中信证券和顺丰控股在日收益率分布两端拥有更多数量的样本，说明极端风险较高



# 第3步：计算每只股票年化的平均收益率、波动率以及协方差等
# 计算股票的年化平均收益率（由于运用的基础数据为日频数据，因此计算结果需要进行年化处理）
R_mean = R.mean() * 252
print(R_mean)

# 计算股票收益率的年化波动率
R_vol = R.std() * np.sqrt(252)
print(R_vol)

# 计算股票的协方差矩阵并且进行年化处理
R_cov = R.cov() * 252
print(R_cov)

# 计算股票的相关系数矩阵
R_corr = R.corr()
print(R_corr)
# 可看出：这5只股票的相关性并不高，因此整个组合的分散化效果较好

# 第4步：根据每只股票配置权重20%计算投资组合年化的预期收益率和波动率
# 投资组合中的个股数量
n = 5

# 投资组合中每只股票相同权重的权重数组
w = np.ones(n) / n

# 查看输出结果
print(w)

# 计算投资组合年化的预期收益率
R_port = np.sum(w * R_mean)
print("投资组合年化的预期收益率", round(R_port, 4))

# 计算投资组合年化的波动率（dot函数）
vol_port = np.sqrt(np.dot(w, np.dot(R_cov, w.T)))
print("投资组合年化的波动率", round(vol_port, 4))



# 投资组合的可行集与有效前沿
# 通过Python绘制可行集
# 沿用上例，针对投资组合配置的5只股票，运用Python随机创建2000个不同的权重数组，并且绘制投资组合的可行集
# 需要创建权重数组的数量
I = 2000

# 创建存放投资组合年化预期收益率的初始数组
Rp_list = np.ones(I)

# 创建存放投资组合年化波动率的初始数组
Vp_list = np.ones(I)

# 通过for语句创建2000个随即权重数组
for i in np.arange(I):
    x = np.random.rand(n)# 从均匀分布中随机抽取0~1的5个随机数
    weights = x / sum(x)# 创建投资组合的随机权重数组
    Rp_list[i] = np.sum(weights * R_mean)# 创建投资组合年化的预期收益率
    Vp_list[i] = np.sqrt(np.dot(weights, np.dot(R_cov, weights.T)))# 计算投资组合年化的波动率

plt.figure(figsize = (9, 6))
plt.scatter(Vp_list, Rp_list)
plt.xlabel(u"波动率", fontsize = 13)
plt.ylabel(u"预期收益率", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"投资组合预期收益率与波动率的关系图", fontsize = 13)
plt.grid()
plt.show()
# 图中的散点就构成投资组合的可行集。
# 在可行集内部，在波动率一定的情况下，理性投资者会选择可行集最上方的点所对应的投资组合，因为可以实现预期收益率的最大化；
# 同样，在预期收益率一定的情况下，理性投资者会选择可行集最左侧的点所对应的投资组合，因为可以实现波动率的最小化，也就是风险最小化
# 综上，理性投资者会选择可行集的包络线所对应的投资组合进行投资，这条包络线就是有效前沿



# 通过Python构建有效前沿（minimize函数）
# 沿用前例信息，同时给定投资组合年化的预期收益率等于15%，运用Python计算使投资组合波动率最小的每只股票配置权重
# 导入SciPy的子模块optimize
import scipy.optimize as sco

# 定义一个求最优值的函数
def f(w):
    w = np.array(w)# 设置投资组合中每只股票的权重
    Rp_opt = np.sum(w * R_mean)# 计算投资组合的预期收益率
    Vp_opt = np.sqrt(np.dot(w, np.dot(R_cov, w.T)))# 计算投资组合的波动率
    return np.array([Rp_opt, Vp_opt])# 以数组格式输出结果

# 定义一个计算最小波动率所对应权重的函数
def Vmin_f(w):
    return f(w)[1]# 输出结果是投资组合的波动率

# 权重的约束条件（以字典格式输入）
cons = ({"type":"eq", "fun":lambda x: np.sum(x) - 1}, {"type":"eq", "fun":lambda x: f(x)[0] - 0.15})

# 权重的边界条件（以元组格式输入）
bnds = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1))

# 创建权重相等的数组作为迭代运算的初始值
w0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# 得到计算的结果
result = sco.minimize(fun = Vmin_f, x0 = w0, method = "SLSQP", bounds = bnds, constraints = cons)

print("投资组合预期收益率15%对应投资组合的波动率", round(result["fun"], 4))
print("投资组合预期收益率15%对应长江电力的权重", round(result["x"][0], 4))
print("投资组合预期收益率15%对应平安银行的权重", round(result["x"][1], 4))
print("投资组合预期收益率15%对应上海机场的权重", round(result["x"][2], 4))
print("投资组合预期收益率15%对应中信证券的权重", round(result["x"][3], 4))
print("投资组合预期收益率15%对应顺丰控股的权重", round(result["x"][-1], 4))
# 此处计算得到的投资组合仅仅是有效前沿的某一个点



# 讨论有效前沿的起点，该点是处于可行集边界上的投资组合波动率全局最小值以及与之相对应的投资组合预期收益率的点
# 沿用上例信息，计算该投资组合波动率的全局最小值、与该最小波动率相对应的预期收益率以及股票的权重
# 设置波动率是全局最小值的约束条件
cons_vmin = ({"type":"eq", "fun":lambda x: np.sum(x) - 1})

# 计算波动率是全局最小值的相关结果
result_vmin = sco.minimize(fun = Vmin_f, x0 = w0, method = "SLSQP", bounds = bnds, constraints = cons_vmin)

# 计算全局最小值的波动率
Vp_vmin = result_vmin["fun"]
print("在可行集上属于全局最小值的波动率", round(Vp_vmin, 4))

# 计算相应的投资组合预期收益率
Rp_vmin = np.sum(R_mean * result_vmin["x"])

print("全局最小值的波动率对应投资组合的预期收益率", round(Rp_vmin, 4))
print("全局最小值的波动率对应长江电力的权重", round(result_vmin["x"][0], 4))
print("全局最小值的波动率对应平安银行的权重", round(result_vmin["x"][1], 4))
print("全局最小值的波动率对应上海机场的权重", round(result_vmin["x"][2], 4))
print("全局最小值的波动率对应中信证券的权重", round(result_vmin["x"][3], 4))
print("全局最小值的波动率对应顺丰控股的权重", round(result_vmin["x"][-1], 4))



# 当投资组合的预期收益率是一个数组时，就可以得到在投资组合波动率最小的情况下，投资组合中每只股票的权重以及对应的投资组合波动率，这些预期收益率和波动率的集合构成了有效前沿
# 沿用前例信息，创建一个以对应波动率全局最小值的投资组合预期收益率作为区间下限，以30%作为区间上限的目标预期收益率等差数组，并计算相对应的波动率数组，从而完成对有效前沿的构建并可视化
# 创建投资组合的目标预期收益率数组
Rp_target = np.linspace(Rp_vmin, 0.3, 200)

# 创建存放对应波动率的初始空列表
Vp_target = []

for r in Rp_target:
    cons_new = ({"type":"eq", "fun":lambda x: np.sum(x) - 1}, {"type":"eq", "fun":lambda x: f(x)[0] - r})# 预期收益率等于目标收益率的约束条件以及股票权重的约束条件
    result_new = sco.minimize(fun = Vmin_f, x0 = w0, method = "SLSQP", bounds = bnds, constraints = cons_new)
    Vp_target.append(result_new["fun"])# 存放每一次计算得到波动率

plt.figure(figsize = (9, 6))
plt.scatter(Vp_list, Rp_list)
plt.plot(Vp_target, Rp_target, "r-", label = u"有效前沿", lw = 2.5)
plt.plot(Vp_vmin, Rp_vmin, "g*", label = u"全局最小波动率", markersize = 13)
plt.xlabel(u"波动率", fontsize = 13)
plt.ylabel(u"预期收益率", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlim(0.15, 0.3)
plt.ylim(0.06, 0.2)
plt.title(u"投资组合的有效前沿", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 可看出：投资组合的有效前沿就是可行集的一条包络线，并且起点就是处于可行集最左端的投资组合波动率全局最小值和与之相对应的投资组合预期收益率的点



# 资本市场线
# 上述资产均为风险资产，现将无风险资产引入投资组合，也就意味着投资者可以按照无风险利率借入或借出资金，其中借入的资金将用于股票投资
# 资产市场线是一条从无风险利率引出的与有效前沿相切的一条切线，并且该切线有且只有一条，切点所对应的投资组合称为市场组合，也称切点组合
# 沿用前例信息，同时无风险利率运用1年期LPR利率，中国人民银行在2020年12月21日公布的该利率报价为3.85%，通过Python测算资本市场线并且可视化
# 第1步：计算资本市场线的斜率、市场组合的预期收益率和波动率
# 1年期LPR利率（无风险利率）
Rf = 0.0385

# 定义一个求最优值的新函数
def F(w):
    w = np.array(w)# 股票的权重
    Rp_opt = np.sum(w * R_mean)# 计算投资组合的预期收益率
    Vp_opt = np.sqrt(np.dot(w, np.dot(R_cov, w.T)))# 计算投资组合的波动率
    Slope = (Rp_opt - Rf) / Vp_opt# 计算资本市场线的斜率
    return np.array([Rp_opt, Vp_opt, Slope])# 以数组格式输出结果

# 定义使负的资本市场线斜率最小化的函数
def Slope_F(w):
    return -F(w)[-1]# 输出结果是负的资本市场线斜率

# 权重的约束条件
cons_Slope = ({"type":"eq", "fun":lambda x: np.sum(x) - 1})
result_Slope = sco.minimize(fun = Slope_F, x0 = w0, method = "SLSQP", bounds = bnds, constraints = cons_Slope)

# 计算资本市场线的斜率
Slope = -result_Slope["fun"]
print("资本市场线的斜率", round(Slope, 4))

# 市场组合的每只股票配置权重
Wm = result_Slope["x"]

print("市场组合配置的长江电力的权重", round(Wm[0], 4))
print("市场组合配置的平安银行的权重", round(Wm[1], 4))
print("市场组合配置的上海机场的权重", round(Wm[2], 4))
print("市场组合配置的中信证券的权重", round(Wm[3], 4))
print("市场组合配置的顺丰控股的权重", round(Wm[-1], 4))

# 市场组合的预期收益率
Rm = np.sum(R_mean * Wm)

# 利用公式计算出市场组合的波动率
Vm = (Rm - Rf) / Slope
print("市场组合的预期收益率", round(Rm, 4))
print("市场组合的波动率", round(Vm, 4))

# 第2步：测算资本市场线并且进行可视化
# 资本市场线的投资组合预期收益率数组
Rp_CML = np.linspace(Rf, 0.25, 200)

# 资本市场线的投资组合波动率数组
Vp_CML = (Rp_CML - Rf) / Slope

plt.figure(figsize = (9, 6))
plt.scatter(Vp_list, Rp_list)
plt.plot(Vp_target, Rp_target, "r-", label = u"有效前沿", lw = 2.5)
plt.plot(Vp_CML, Rp_CML, "b--", label = u"资本市场线", lw = 2.5)
plt.plot(Vm, Rm, "y*", label = u"市场组合", markersize = 14)
plt.plot(Vp_vmin, Rp_vmin, "g*", label = u"全局最小波动率", markersize = 14)
plt.xlabel(u"波动率", fontsize = 13)
plt.ylabel(u"预期收益率", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlim(0.0, 0.3)
plt.ylim(0.03, 0.22)
plt.title(u"投资组合理论的可视化", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 可看出：资本市场线上的任意一点都可以由无风险资产与市场组合构建的一个投资组合来表示。
# 此外，在位于市场组合左侧的资本市场线上，投资组合表示投资者将自有资金按照一定比例配置了无风险资产，剩余比例配置了市场组合；
# 在位于市场组合右侧的资本市场线上，投资组合则表示除了投资者的自有资金外，投资者还按照无风险利率融资并将其全部投资于市场组合，即运用了杠杆投资



# 资本资产定价模型
# 一个金融机构希望运用上证180指数的成分股模拟投资组合，考察投资组合中不同股票数量与整个投资组合波动率（风险）之间的关系。假定选择2018年至2020年作为观察期间，
# 按照交易日进行观测。鉴于成分股中的少数上市公司是在观测期内才上市的股票剔除，最终保留160只股票
# 在投资组合中逐次增加成分股股票数量（N）并且确保投资组合中的不同股票有相同权重（1/N）。比如，第1次仅配置1只股票，权重为100%；第2次配置2只股票，每只股票权重为50%；
# 第3次配置3只股票，每只股票权重降至1/3，以此类推，一直到最后第160次配置全部160只股票，每只股票权重为1/160。
# 第1步：从外部导入160只股票2018年至2020年的收盘价数据
# 从外部导入数据
# TODO: price_stocks = pd.read_excel(, sheet_name = "Sheet1", header = 0, index_col = 0)

# 查看数据框的列名
print(price_stocks.column)

# 查看数据框的行索引
print(price_stocks.index)
# 可看出：一共有160只股票以及730个交易日的收盘价数据

# 第2步：计算每只股票的日收益率数据，并且运用for语句快速计算不同股票数量对应的投资组合波动率
# 建立股票的日收益率时间序列
return_stocks = np.log(price_stocks / price_stocks.shift(1))

# 计算得到股票的数量
n = len(return_stocks.columns)

# 创建存放投资组合波动率的初始数组
vol_port = np.zeros(n)

for i in range(1, n+1):
    w = np.ones(i) / i# 逐次计算股票的等权重数组
    cov = 252 * return_stocks.iloc[:, i].cov()# 逐次计算不同股票之间的年化协方差
    vol_port[i-1] = np.sqrt(np.dot(w, np.dot(cov, w.T)))# 逐次计算不同股票之间的年化波动率

# 第3步：运用在第2步计算得到的针对不同股票数量所对应的投资组合波动率，可视化股票数量与投资波动率之间的关系
# 创建1~160的整数数组
N_list = np.arange(n) + 1

plt.figure(figsize = (9, 6))
plt.plot(N_list, vol_port, "r-", lw = 2.0)
plt.xlabel(u"投资组合中的股票数量", fontsize = 13)
plt.ylabel(u"投资组合波动率", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"投资组合中的股票数量与投资组合波动率之间的关系图", fontsize = 13)
plt.grid()
plt.show()
# 可看出：随着投资组合配置的股票数量N不断增加，投资组合波动率刚开始的时候是迅速下降，但当整个投资组合超过40只的时候，整体的投资组合波动率开始趋于稳定，且基本保持在23%的水平。
# 可得出：1.当投资组合配置的股票超过40只时，投资组合的风险很接近系统风险；2.波动率23%可以视作系统风险，单只股票的波动率超过23%就属于非系统风险



# 模型数学表达式及运用
# 模型数学表达式与Python自定义函数
def Ri_CAPM(beta, Rm, Rf):
    """
    定义一个运用资本资产定价模型计算股票预期收益率的函数
    beta：代表股票的贝塔值
    Rm：代表市场收益率
    Rf：代表无风险利率
    """
    Ri = Rf + beta * (Rm - Rf)# 资本资产定价模型的表达式
    return Ri



# 计算贝塔值和预期收益率
# 金融机构的证券分析师希望计算招商银行A股的贝塔值，并且以沪深300指数作为市场组合，基于2017年至2020年的日收盘价数据计算股票的贝塔值。
# 此外，无风险收益率依然运用2020年12月21日中国人民银行公布的1年期LPR利率，该利率为3.85%
# 第1步：导入招商银行A股和沪深300指数2017年至2020年期间的日收盘价数据，并且计算日收益率的时间序列
# 导入外部数据
P_bank_index = pd.read_excel("C:/Users/InsaneHe/desktop/Python/招商银行A股与沪深300指数日收盘价数据（2017-2020年）.xlsx",
                             sheet_name = "Sheet1", header = 0, index_col = 0)

# 按照对数收益率的计算公式得到日收益率的数据框
R_bank_index = np.log(P_bank_index / P_bank_index.shift(1))

# 删除默认的数据
R_bank_index = R_bank_index.dropna()

# 查看描述性统计
print(R_bank_index.describe())
# 可看出：观测期内一共有973个交易日；无论是招商银行A股还是沪深300指数，平均日收益率均为正，并且招商银行A股收益率是沪深300指数的两倍

# 第2步：计算招商银行A股的贝塔值，需要运用前面介绍的statsmodels的子模块api
# 导入statsmodels的子模块api
import statsmodels.api as sm

# 取招商银行A股的日收益率序列（因变量）
R_bank = R_bank_index["招商银行"]

# 取沪深300指数的日收益率序列（自变量）
R_index = R_bank_index["沪深300指数"]

# 对自变量的样本值增加一列常数项
R_index_addcons = sm.add_constant(R_index)

# 构建普通最小二乘法的线性回归模型
model = sm.OLS(endog = R_bank, exog = R_index_addcons)

# 拟合线性回归模型
result = model.fit()

print(result.summary())
print(result.params)
# 可看出：招商银行A股的贝塔值是0.958032，阿尔法值是0.000482

# 第3步：利用前面的自定义函数Ri_CAPM，计算招商银行A股的预期收益率
# 1年期LPR利率（无风险利率）
LPR_1Y = 0.0385

# 计算沪深300指数的年化收益率
R_market = 252 * R_index.mean()

# 计算招商银行A股的预期收益率（年化）
R_stock = Ri_CAPM(beta = result.params[-1], Rm = R_market, Rf = LPR_1Y)
print("招商银行A股的年化预期收益率", round(R_stock, 6))



# 证券市场线
# 沿用前例的信息和计算结果，同时对招商银行A股的贝塔值取[0, 2.0]区间的等差数列，计算对应于不同贝塔值的股票预期收益率，并且进行可视化
# 设定一个贝塔值的数组
beta_list = np.linspace(0, 2.0, 100)

# 计算招商银行A股预期收益率
R_stock_list = Ri_CAPM(beta = beta_list, Rm = R_market, Rf = LPR_1Y)

plt.figure(figsize=(9, 6))
plt.plot(beta_list, R_stock_list, "r-", label = u"证券市场线", lw = 2.5)
plt.plot(result.params[-1], R_stock, "o", lw = 2.5)
plt.axis("tight")
plt.xticks(fontsize = 13)
plt.xlabel(u"贝塔值", fontsize = 13)
plt.xlim(0, 2.0)
plt.yticks(fontsize = 13)
plt.ylabel(u"股票预期收益率", fontsize = 13)
plt.ylim(0, 0.2)
plt.title(u"资本资产定价模型（以招商银行A股为例）", fontsize = 13)
plt.annotate(u"贝塔值等于0.958对应的收益率", fontsize = 14, xy = (0.96, 0.1115), xytext = (1.0, 0.06),
             arrowprops = dict(facecolor = "b", shrink = 0.05))
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 可看出：证券市场线的截距就是无风险利率，斜率为特雷诺比率



# 投资组合的绩效评估
# 衡量投资组合的管理是否成熟和规范的核心标准就是考察投资组合实际承担的风险与获取的收益之间是否匹配
# 金融机构配置了4只开放式股票型基金，分别是中海量化策略混合基金、南方前例新蓝筹混合基金、交银精选混合基金以及天弘惠利灵活配置混合基金



# 夏普比率
# 指在某一时间区间内，投资组合的收益率高于无风险利率的部分除以投资组合波动率之后所得到的比值，反映了投资组合承担每一单位风险所带来的超额收益（以无风险利率作为比较基准）
# 实际即为资本市场线的斜率
# 通过Python自定义一个计算夏普比率的函数
def SR(Rp, Rf, Vp):
    """
    定义一个计算夏普比率的函数
    Rp：代表投资组合的年化收益率
    Rf：代表无风险利率
    Vp：代表投资组合的年化波动率
    """
    sharp_ratio = (Rp - Rf) / Vp
    return sharp_ratio

# 沿用上例，同时无风险利率选择商业银行1年期存款的基准利率并且等于1.5%，计算中海量化策略混合基金、南方潜力新蓝筹混合基金、交银精选混合基金以及天弘惠利灵活配置混合基金这4只基金的夏普比率
# 第1步：导入外部数据并且绘制每只基金净值的日走势图
# 导入外部数据
fund = pd.read_excel("D:\Python\国内4只开放式股票型基金净值数据（2018-2020）.xlsx",
                     sheet_name = "Sheet1", header = 0, index_col = 0)

# 基金净值可视化
fund.plot(figsize = (9, 6), grid = True)
# 可看出：天弘惠利灵活配置混合基金的净值走势比较稳健，其他3只基金的净值波动较大并且在走势上存在趋同性

# 第2步：计算2018年至2020年基金的年化收益率和波动率，同时，运用前面的自定义函数SR计算这4只基金的夏普比率
# 创建基金日收益率的时间序列
R_fund = np.log(fund / fund.shift(1))

# 删除缺失值
R_fund = R_fund.dropna()

# 计算全部3年的平均年化收益率
R_mean = R_fund.mean() * 252

# 计算全部3年的年化波动率
Sigma = R_fund.std() * np.sqrt(252)

# 1年期银行存款基准利率作为无风险利率
R_f = 0.015

SR_3years = SR(Rp = R_mean, Rf = R_f, Vp = Sigma)
print("2018年至2020年3年平均的夏普比率\n", round(SR_3years, 4))
# 可看出：按照夏普比率排名：天弘惠利灵活配置混合基金排名第1，中海量化策略混合基金则排名倒数第一

# 第3步：计算2018年至2020年期间每年的夏普比率
# 获取2018年的日收益率
R_fund2018 = R_fund.loc["2018-01-01":"2018-12-31"]

# 获取2019年的日收益率
R_fund2019 = R_fund.loc["2019-01-01":"2019-12-31"]

# 获取2020年的日收益率
R_fund2020 = R_fund.loc["2020-01-01":"2020-12-31"]

# 计算2018年的年化收益率
R_mean_2018 = R_fund2018.mean() * 252

# 计算2019年的年化收益率
R_mean_2019 = R_fund2019.mean() * 252

# 计算2020年的年化收益率
R_mean_2020 = R_fund2020.mean() * 252

# 计算2018年的年化波动率
Sigma_2018 = R_fund2018.std() * np.sqrt(252)

# 计算2019年的年化波动率
Sigma_2019 = R_fund2019.std() * np.sqrt(252)

# 计算2020年的年化波动率
Sigma_2020 = R_fund2020.std() * np.sqrt(252)

# 计算2018年的夏普比率
SR_2018 = SR(Rp = R_mean_2018, Rf = R_f, Vp = Sigma_2018)
print("2018年的夏普比率\n", round(SR_2018, 4))

# 计算2019年的夏普比率
SR_2019 = SR(Rp = R_mean_2019, Rf = R_f, Vp = Sigma_2019)
print("2019年的夏普比率\n", round(SR_2019, 4))

# 计算2020年的夏普比率
SR_2020 = SR(Rp = R_mean_2020, Rf = R_f, Vp = Sigma_2020)
print("2020年的夏普比率\n", round(SR_2020, 4))



# 索提诺比率
# 分子与夏普比率相同，分母为下行标准差，反映了投资组合承担每一单位下行风险所带来的超额收益（以无风险利率作为比较基准）
# 通过Python自定义一个计算索提诺比率的函数
def SOR(Rp, Rf, Vd):
    """
    定义一个计算索提诺比率的函数
    Rp：表示投资组合的年化收益率
    Rf：表示无风险利率
    Vd：表示投资组合收益率的年化下行标准差
    """
    sortino_ratio = (Rp - Rf) / Vd
    return sortino_ratio

# 沿用上例，同时无风险利率依然选择商业银行1年期存款基准利率1.5%，计算中海量化策略混合基金、南方潜力新蓝筹混合基金、交银精选混合基金以及天弘惠利灵活配置混合基金这4只基金的索提诺比率
# 第1步：计算每只基金收益率的下行标准差
# 创建放置基金收益率下行标准差的初始数组
V_down = np.zeros_like(R_mean)

for i in range(len(V_down)):
    R_neg = R_fund.iloc[:, i][R_fund.iloc[:, i]<0]# 生成基金收益率为负的时间序列
    N_down = len(R_neg)# 计算亏损的交易日天数
    V_down[i] = np.sqrt(252) * np.sqrt(np.sum(R_neg ** 2) / N_down)# 计算年化下行标准差
    print(R_fund.columns[i], "年化下行标准差", round(V_down[i], 4))

# 第2步：运用前面的自定义函数SOR计算每只基金2018年至2020年3年平均1索提诺比率
# 计算索提诺比率
SOR_3years = SOR(Rp = R_mean, Rf = R_f, Vd = V_down)
print("2018年至2020年3年平均的索提诺比率\n", round(SOR_3years, 4))
# 通常而言，若收益率的分布是左偏的，即出现亏损的样本数量多于盈利的样本数量，则相对于夏普比率，索提诺比率会更加合适。若在某个观测期内投资组合收益率为负数的样本数量很少甚至没有，则会影响索提诺比率的使用



# 特雷诺比率
# 分子与夏普比率、索提诺比率相比，分子均相同，分母调整为投资组合的贝塔值
# 表示当投资组合每承受一单位系统风险时，会产生多少的风险溢价（以无风险利率作为比较基准）（即证券市场线的斜率）
# 当投资组合的非系统风险已被有效分散时，只考虑系统风险，特雷诺比率就相对更加合适
# 通过Python自定义一个计算特雷诺比率的函数
def TR(Rp, Rf, beta):
    """
    定义一个计算特雷诺比率的函数
    Rp：表示投资组合的年化收益率
    Rf：表示无风险利率
    beta：表示投资组合的贝塔值
    """
    treynor_ratio = (Rp - Rf) / beta# 特雷诺比率的数学表达式
    return treynor_ratio

# 沿用上例，同时将沪深300指数作为市场组合，依次计算中海量化策略混合基金、南方潜力新蓝筹混合基金、交银精选混合基金以及天弘惠利灵活配置混合基金这4只基金的贝塔值，并最终计算基金的特雷诺比率
# 第1步：导入沪深300指数的数据，同时为了计算的高效，运用for语句快速计算每只基金的贝塔值
# 导入沪深300指数的数据
HS300 = pd.read_excel("D:\Python\沪深300指数日收盘价数据（2018-2020年）.xlsx", sheet_name = "Sheet1",
                      header = 0, index_col = 0)

# 创建沪深300指数的日收益率序列
R_HS300 = np.log(HS300 / HS300.shift(1))

# 删除缺失值
R_HS300 = R_HS300.dropna()

# 沪深300指数日收益率序列（自变量）增加一列常数项
X_addcons = sm.add_constant(R_HS300)

# 创建放置基金贝塔值的初始数组
betas = np.zeros_like(R_mean)

# 创建放置线性回归方程常数项的初始数组
cons = np.zeros_like(R_mean)

for i in range(len(R_mean)):
    Y = R_fund.iloc[:, i]# 设定因变量的样本值
    model = sm.OLS(endog = Y, exog = X_addcons)# 构建普通最小二乘法的线性回归模型
    result = model.fit()# 创建一个线性回归的结果对象
    cons[i] = result.params[0]# 逐一存放线性回归方程常数项
    betas[i] = result.params[1]# 逐一存放基金的贝塔值
    print(R_fund.columns[i], "贝塔值", round(betas[i], 4))

# 第2步：将线性回归的结果进行可视化
# 创建对应x轴的数组
X_list = np.linspace(np.min(R_HS300), np.max(R_HS300), 200)

plt.figure(figsize = (11, 10))

# 逐一绘制基金与指数的散点图和拟合的线性回归
for i in range(len(R_mean)):
    plt.subplot(2, 2, i+1)
    plt.scatter(R_HS300, R_fund.iloc[:, i])
    plt.plot(X_list, cons[i] + betas[i] * X_list, "r-", label = u"线性回归拟合", lw = 2.0)
    plt.xlabel(u"沪深300指数", fontsize = 13)
    plt.ylabel(R_fund.columns[i], fontsize = 13)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.legend(fontsize = 13)
    plt.grid()
plt.show()
# 可看出：针对每只基金的收益率而言，将沪深300指数作为市场组合所得到的线性回归拟合程度时比较高的

# 第3步：根据前面的自定义函数TR，计算每只基金2018年至2020年3年平均的特雷诺比率
# 计算特雷诺比率
TR_3years = TR(Rp = R_mean, Rf = R_f, beta = betas)
print("2018年至2020年3年平均的特雷诺比率\n", round(TR_3years, 4))



# 卡玛比率
# 描述收益率和最大回撤率之间的关系，等于年化收益率与历史最大回撤率的比值
# 表示承担每一个单位的回撤风险，可以获得多少绝对收益的补偿
# 最大回撤率：在选定的交易期内，任一交易时点往后推算，投资组合市值或者基金净值触及最低点时的收益率回撤幅度的最大值（用于描述投资组合可能出现的最糟糕情况）
# 最大回撤率即为全部的期间回撤率数据中的最大值
# 通过Python自定义一个计算卡玛比率的函数
def CR(Rp, MDD):
    """
    定义一个计算卡玛比率的函数
    Rp：表示投资组合的年化收益率
    MDD：表示投资组合的最大回撤率
    """
    calmar_ratio = Rp / MDD# 卡玛比率的数学表达式
    return calmar_ratio

# 通过Python自定义一个计算投资组合最大回撤率的函数
def MDD(data):
    """
    定义一个计算投资组合（以基金为例）最大回撤率的函数
    data：代表某只基金的净值数据，以序列或者数据框格式输入
    """
    N = len(data)# 计算期间的交易日天数
    DD = np.zeros((N-1, N-1))# 创建元素为0的N-1行，N-1列数组，用于存放回撤率数据
    for i in range(N-1):# 第1个for语句
        Pi = data.iloc[i]# 第i个交易日的基金净值
        for j in range(i+1, N):# 第2个for语句
            Pj = data.iloc[j]# 第j个交易日的基金净值
            DD[i, j-1] = (Pi - Pj) / Pi# 依次计算并存放期间的每个回撤率数据
    Max_DD = np.max(DD)# 计算基金净值的最大回撤率
    return Max_DD

# 沿用上例，计算中海量化策略混合基金、南方潜力新蓝筹混合基金、交银精选混合基金以及天弘惠利灵活配置混合基金这4只基金在2018年至2020年期间的最大回撤率，并最终计算基金的卡玛比率
# 第1步：根据自定义函数MDD，依次计算4只基金的最大回撤率
# 选取中海量化策略基金净值的时间序列
fund_zhonghai = fund["中海量化策略基金"]

# 选取南方新蓝筹基金净值的时间序列
fund_nanfang = fund["南方新蓝筹基金"]

# 选取交银精选混合基金
fund_jiaoyin = fund["交银精选基金"]

# 选取天弘惠利基金净值的时间序列
fund_tianhong = fund["天弘惠利基金"]

# 计算中海量化策略基金的最大回撤率
MDD_zhonghai = MDD(data = fund_zhonghai)

# 计算南方新蓝筹基金的最大回撤率
MDD_nanfang = MDD(data = fund_nanfang)

# 计算交银精选基金的最大回撤率
MDD_jiaoyin = MDD(data = fund_jiaoyin)

# 计算天弘惠利基金的最大回撤率
MDD_tianhong = MDD(data = fund_tianhong)

print("2018年至2020年中海量化策略基金的最大回撤率", round(MDD_zhonghai, 4))
print("2018年至2020年南方新蓝筹基金的最大回撤率", round(MDD_nanfang, 4))
print("2018年至2020年交银精选基金的最大回撤率", round(MDD_jiaoyin, 4))
print("2018年至2020年天弘惠利基金的最大回撤率", round(MDD_tianhong, 4))

# 第2步：根据自定义函数CR，依次计算4只基金的卡玛比率
# 计算中海量化策略基金的卡玛比率
CR_zhonghai = CR(Rp = R_mean["中海量化策略基金"], MDD = MDD_zhonghai)

# 计算南方新蓝筹基金的卡玛比率
CR_nanfang = CR(Rp = R_mean["南方新蓝筹基金"], MDD = MDD_nanfang)

# 计算交银精选基金的卡玛比率
CR_jiaoyin = CR(Rp = R_mean["交银精选基金"], MDD = MDD_jiaoyin)

# 计算天弘惠利基金的卡玛比率
CR_tianhong = CR(Rp = R_mean["天弘惠利基金"], MDD = MDD_tianhong)

print("2018年至2020年中海量化策略基金的卡玛比率", round(CR_zhonghai, 4))
print("2018年至2020年南方新蓝筹基金的卡玛比率", round(CR_nanfang, 4))
print("2018年至2020年交银精选基金的卡玛比率", round(CR_jiaoyin, 4))
print("2018年至2020年天弘惠利基金的卡玛比率", round(CR_tianhong, 4))
# 卡玛比率用最大回撤率评估风险，更便于投资者理解，但是忽略了投资组合的总体波动性



# 信息比率
# 为了能使绩效评估更加公允，需要引入一个用于对比的参照系或者比较基准，这个参照系为基准组合，通常以证券市场广泛使用的股票指数作为基准组合
# 跟踪偏离度：投资组合收益率与基准组合收益率之间的差异；跟踪误差：投资组合收益率与基准组合收益率之间差异的标准差，实质为跟踪偏离度的标准差，衡量投资组合的主动管理风险
# 跟踪误差：投资组合收益率与基准组合收益率之间差异的标准差，实质为跟踪偏离度的标准差，衡量投资组合的主动管理风险
# 信息比率：跟踪偏离度与跟踪误差的比率，从主动管理的角度描述投资组合风险调整后的收益，衡量投资组合承担主动管理风险所带来的超额收益（相对于基准组合）
# 通过Python自定义一个计算信息比率的函数
def IR(Rp, Rb, TE):
    """
    定义一个计算信息比率的函数
    Rp：表示投资组合的年化收益率
    Rb：表示基准组合的年化收益率
    TE：表示跟踪误差
    """
    information_ratio = (Rp - Rb) / TE
    return information_ratio

# 沿用前例，以沪深300指数作为基准组合，计算中海量化策略混合基金、南方潜力新蓝筹混合基金、交银精选混合基金以及天弘惠利灵活配置混合基金这4只基金在2018年至2020年期间的信息比率
# 第1步：计算每只基金的跟踪误差，为了提高计算效率，需要运用for语句
# 创建存放基金跟踪误差的初始数组
TE_fund = np.zeros_like(R_mean)

for i in range(len(R_mean)):
    TD = np.array(R_fund.iloc[:, i]) - np.array(R_HS300.iloc[:, 0])# 计算基金跟踪偏离度并以数组格式存放
    TE_fund[i] = TD.std() * np.sqrt(252)# 计算并存放每只基金的年化跟踪误差
    print(R_fund.columns[i], "跟踪误差", round(TE_fund[i], 4))

# 第2步：利用自定义函数IR计算每只基金的信息比率
# 计算沪深300指数的年化收益率
R_mean_HS300 = R_HS300.mean() * 252

# 转换成浮点型
R_mean_HS300 = float(R_mean_HS300)

IR_3years = IR(Rp = R_mean, Rb = R_mean_HS300, TE = TE_fund)
print("2018年至2020年3年平均的信息比率\n", round(IR_3years, 4))
