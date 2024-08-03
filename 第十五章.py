# 第十五章
# 15.1.2 风险价值的可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["FangSong"]
mpl.rcParams["axes.unicode_minus"] = False
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# 导入ScciPy的子模块stats
import scipy.stats as st

# 设定95%的置信水平
x = 0.95

# 计算正态分布的分位数
z = st.norm.ppf(q = 1-x)

# 创建从-4到4的等差数列（投资组合盈亏）
x = np.linspace(-4, 4, 200)

# 计算正态分布的概率密度函数值
y = st.norm.pdf(x)

# 创建从-4到z的等差数列
x1 = np.linspace(-4, z, 100)

# 计算正态分布的概率密度函数值
y1 = st.norm.pdf(x1)

plt.figure(figsize=(9, 6))
plt.plot(x, y, "r-", lw = 2.0)
plt.fill_between(x1, y1)# 颜色填充
plt.xlabel(u"投资组合盈亏", fontsize = 13)
plt.ylabel(u"概率密度",fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylim(0, 0.42)
plt.annotate("VaR", xy = (z-0.02, st.norm.pdf(z) + 0.005), xytext = (-2.3, 0.17),
             arrowprops = dict(shrink = 0.01), fontsize = 13)# 绘制箭头
plt.title(u"假定投资组合盈亏服从正态分布的风险价值（VaR）", fontsize = 13)
plt.grid()
plt.show()



# 15.2 方差-协方差法
# 通过Python自定义一个运用方差-协方差法计算风险价值的函数
def VaR_VCM(Value, Rp, Vp, X, N):
    """
    定义一个运用方差-协方差法计算风险价值的函数
    Value：代表投资组合的价值或市值
    Rp：代表投资组合日平均收益率
    Vp：投资组合收益率的日波动率
    X：代表置信水平
    N：代表持有期，用天数表示
    """
    import scipy.stats as st # 导入SciPy的子模块stats
    from numpy import sqrt # 从NumPy模块中导入函数sqrt
    z = abs(st.norm.ppf(q = 1-X)) # 计算标准正态分布下1-X的分位数并取绝对值
    VaR_1day = Value * (z * Vp - Rp) # 计算持有期为1天的风险价值
    VaR_Nday = sqrt(N) * VaR_1day # 计算持有期为N天的风险价值
    return VaR_Nday



# 15.2.2 方差-协方差法的应用
# 见P583
# 第1步：导入外部数据并且计算每个资产的日平均收益率、日波动率等
price = pd.read_excel("D:/Python/投资组合配置资产的每日价格（2018年至2020年）.xlsx", sheet_name = "Sheet1", header = 0, index_col = 0)
price = price.dropna() # 删除缺失值
price.index = pd.DatetimeIndex(price.index) # 将数据框行索引转为datetime格式
(price / price.iloc[0]).plot(figsize = (9, 6), grid = True) # 将首个交易日价格归一并且可视化
# 可看出：在2018年至2020年期间，投资组合中只有贵州茅台、博时标普500ETF基金这2个资产的价格保持上涨，
# 其余3个资产的价格或下跌或“原地踏步”

R = np.log(price / price.shift(1)) # 计算对数收益率
R = R.dropna() # 删除缺失值
R.describe() # 显示描述性统计指标

R_mean = R.mean() # 计算每个资产的日平均收益率
print("2018年至2020年期间日平均收益率\n", R_mean)

R_vol = R.std() # 计算每个资产收益率的日波动率
print("2018年至2020年期间日波动率\n", R_vol)

R_cov = R.cov() # 计算每个资产收益率之间的协方差矩阵

R_corr = R.corr() # 计算每个资产收益率之间的相关系数矩阵
print(R_corr) # 输出相关系数矩阵
# 可看出：输出的不同资产收益率之间相关系数也比较低，说明投资组合资产配置的风险分散化效果较好

# 第2步：按照投资组合当前每个资产的权重计算投资组合的日平均收益率和日波动率
W = np.array([0.15, 0.20, 0.50, 0.05, 0.10]) # 投资组合中各资产配置的权重

Rp_daily = np.sum(W * R_mean) # 计算投资组合日平均收益率
print("2018年至2020年期间投资组合的日平均收益率", round(Rp_daily, 6))

Vp_daily = np.sqrt(np.dot(W, np.dot(R_cov, W.T))) # 计算投资组合日波动率
print("2018年至2020年期间投资组合的日波动率", round(Vp_daily, 6))
# 可看出：2018年至2020年期间投资组合的日平均收益率为正，但是投资组合的日波动率远高于日平均收益率

# 第3步：运用自定义函数VaR_VCM，计算方差-协方差法测算的风险价值
value_port = 1e10 # 投资组合的最新市值为100亿元
D1 = 1 # 持有期为1天
D2 = 10 # 持有期为10天
X1 = 0.95 # 置信水平为95%
X2 = 0.99 # 置信水平为99%

VaR95_1day_VCM = VaR_VCM(Value = value_port, Rp = Rp_daily, Vp = Vp_daily, X = X1, N = D1) # 持有期为1天，置信水平为95%的风险价值

VaR99_1day_VCM = VaR_VCM(Value = value_port, Rp = Rp_daily, Vp = Vp_daily, X = X2, N = D1) # 持有期为1天，置信水平为99%的风险价值

print("方差-协方差法计算持有期为1天、置信水平为95%的风险价值", round(VaR95_1day_VCM, 2))
print("方差-协方差法计算持有期为1天、置信水平为99%的风险价值", round(VaR99_1day_VCM, 2))

VaR95_10day_VCM = VaR_VCM(Value = value_port, Rp = Rp_daily, Vp = Vp_daily, X = X1, N = D2) # 持有期为10天，置信水平为95%的风险价值

VaR99_10day_VCM = VaR_VCM(Value = value_port, Rp = Rp_daily, Vp = Vp_daily, X = X2, N = D2) # 持有期为10天，置信水平为95%的风险价值

print("方差-协方差法计算持有期为10天、置信水平为95%的风险价值", round(VaR95_10day_VCM, 2))
print("方差-协方差法计算持有期为10天、置信水平为99%的风险价值", round(VaR99_10day_VCM, 2))



# 15.3 历史模拟法
# 15.3.2 历史模拟法的运用
# 沿用上例投资组合信息，并运用历史模拟法计算持有期分别为1天和10天、置信水平分别为95%和99%条件下的风险价值
# 第1步：依据2018年至2020年相关资产的日收益率数据，同时结合2020年12月31日投资组合的最新市值和每个资产在投资组合中的最新权重，
# 模拟出2018年至2020年每个交易日投资组合的日收益金额数据并可视化
value_past = value_port * W # 用投资组合最新市值和资产权重计算每个资产最新市值

profit_past = np.dot(R, value_past) # 2018年至2020年每个交易日投资组合模拟盈亏金额
profit_past = pd.DataFrame(data = profit_past, index = R.index, columns = ["投资组合的模拟日收益"]) # 转换为数据框

profit_past.plot(figsize = (9, 6), grid = True) # 将投资组合的模拟日收益可视化

# 第2步：对投资组合的模拟日收益金额进行正态性检验
plt.figure(figsize = (9, 6))
plt.hist(np.array(profit_past), bins = 30, facecolor = "y", edgecolor = "k") # 绘制投资组合的模拟日收益金额直方图并在输入时将数据框转换为数组
plt.xticks(fontsize = 13)
plt.xlabel(u"投资组合的模拟日收益金额", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylabel(u"频数", fontsize = 13)
plt.title(u"投资组合模拟日收益金额的直方图", fontsize = 13)
plt.grid()
plt.show()

st.kstest(rvs = profit_past["投资组合的模拟日收益"], cdf = "norm") # Kolmogorov-Smirnov检验

st.anderson(x = profit_past["投资组合的模拟日收益"], dist = "norm") # Anderson-Darling检验

st.shapiro(profit_past["投资组合的模拟日收益"]) # Shapiro-Wilk检验

st.normaltest(profit_past["投资组合的模拟日收益"]) # 一般的正态性检验

# 第3步：计算投资组合的风险价值
VaR95_1day_history = np.abs(profit_past.quantile(q = 1-X1)) # 持有期为1天、置信水平为95%的风险价值
VaR99_1day_history = np.abs(profit_past.quantile(q = 1-X2)) # 持有期为1天、置信水平为99%的风险价值
VaR95_1day_history = float(VaR95_1day_history) # 转换为浮点型数据
VaR99_1day_history = float(VaR99_1day_history)
print("历史模拟法计算持有期为1天、置信水平为95%的风险价值", round(VaR95_1day_history, 2))
print("历史模拟法计算持有期为1天、置信水平为99%的风险价值", round(VaR99_1day_history, 2))

VaR95_10day_history = np.sqrt(D2) * VaR95_1day_history # 持有期为10天、置信水平为95%的风险价值
VaR99_10day_history = np.sqrt(D2) * VaR99_1day_history # 持有期为10天、置信水平为99%的风险价值
print("历史模拟法计算持有期为10天、置信水平为95%的风险价值", round(VaR95_10day_history, 2))
print("历史模拟法计算持有期为10天、置信水平为99%的风险价值", round(VaR99_10day_history, 2))



# 15.4 蒙特卡罗模拟法
# 15.4.2 蒙特卡罗模拟法的运用
# 沿用前例，通过对贵州茅台、交通银行、嘉实增强信用基金、华夏恒生ETF基金、博时标普500ETF基金这5个资产在下一个交易日的价格或净值进行10万次模拟，
# 进而求出持有期分别为1天和10天、置信水平依次为95%和99%的投资组合风险价值
"""
分别运用学生t分布和正态分布作为资产收益率服从的分布。股票等金融资产的收益率服从学生t分布时，自由度估计值通常处于[4, 8]区间，此处将学生t分布的自由度设定为8
"""
# 第1步：输入相关参数，并运用公式模拟得到投资组合中每个资产在下一个交易日的价格
import numpy.random as npr # 导入NumPy的子模块random

I = 100000 # 模拟的次数
n = 8 # 学生t分布的自由度
epsilon = npr.standard_t(df = n, size = I) # 从学生t分布进行抽样

P1 = price.iloc[-1, 0] # 投资组合中第1个资产（贵州茅台）最新收盘价
P2 = price.iloc[-1, -1] # 投资组合中第2个资产（交通银行）最新收盘价
P3 = price.iloc[-1, 2] # 投资组合中第3个资产（嘉实增强信用基金）最新基金净值
P4 = price.iloc[-1, 3] # 投资组合中第4个资产（华夏恒生ETF基金）最新基金净值
P5 = price.iloc[-1, -1] # 投资组合中第5个资产（博时标普500ETF基金）最新基金净值

R_mean = R.mean() * 252 # 每个资产的年化平均收益率
R_vol = R.std() * np.sqrt(252) # 每个资产收益率的年化波动率
dt = 1 / 252 # 设定步长为一个交易日

P1_new = P1 * np.exp((R_mean[0] - 0.5 * R_vol[0] ** 2) * dt + R_vol[0] * epsilon * np.sqrt(dt)) # 模拟投资组合中第1个资产下一个交易日的收盘价
P2_new = P2 * np.exp((R_mean[1] - 0.5 * R_vol[1] ** 2) * dt + R_vol[1] * epsilon * np.sqrt(dt)) # 模拟投资组合中第2个资产下一个交易日的收盘价
P3_new = P3 * np.exp((R_mean[2] - 0.5 * R_vol[2] ** 2) * dt + R_vol[2] * epsilon * np.sqrt(dt)) # 模拟投资组合中第3个资产下一个交易日的收盘价
P4_new = P4 * np.exp((R_mean[3] - 0.5 * R_vol[3] ** 2) * dt + R_vol[3] * epsilon * np.sqrt(dt)) # 模拟投资组合中第4个资产下一个交易日的收盘价
P5_new = P5 * np.exp((R_mean[-1] - 0.5 * R_vol[-1] ** 2) * dt + R_vol[-1] * epsilon * np.sqrt(dt)) # 模拟投资组合中第5个资产下一个交易日的收盘价

# 第2步：模拟单个资产和整个投资组合在下一个交易日的收益并且可视化
profit1 = (P1_new / P1 - 1) * value_port * W[0] # 模拟第1个资产下一个交易日的收益
profit2 = (P2_new / P2 - 1) * value_port * W[1] # 模拟第2个资产下一个交易日的收益
profit3 = (P3_new / P3 - 1) * value_port * W[2] # 模拟第3个资产下一个交易日的收益
profit4 = (P4_new / P4 - 1) * value_port * W[3] # 模拟第4个资产下一个交易日的收益
profit5 = (P5_new / P5 - 1) * value_port * W[-1] # 模拟第5个资产下一个交易日的收益

profit_port = profit1 + profit2 + profit3 + profit4 + profit5 # 整个投资组合下一个交易日的收益

plt.figure(figsize = (9, 6))
plt.hist(profit_port, bins = 50, facecolor = "y", edgecolor = "k") # 投资组合模拟日收益金额的直方图
plt.xticks(fontsize = 13)
plt.xlabel(u"投资组合模拟的日收益金额", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylabel(u"频数", fontsize = 13)
plt.title(u"通过蒙特卡罗模拟（服从学生t分布）得到投资组合日收益金额的直方图", fontsize = 13)
plt.grid()
plt.show()

# 第3步：运用蒙特卡罗模拟法并且假定资产收益率服从学生t分布的情况下，计算投资组合的风险价值
VaR95_1day_MCst = np.abs(np.percentile(a = profit_port, q = (1-X1) * 100)) # 持有期为1天、置信水平为95%的风险价值
VaR99_1day_MCst = np.abs(np.percentile(a = profit_port, q = (1-X2) * 100)) # 持有期为1天、置信水平为99%的风险价值

print("蒙特卡罗模拟法（服从学生t分布）计算持有期为1天、置信水平为95%的风险价值", round(VaR95_1day_MCst, 2))
print("蒙特卡罗模拟法（服从学生t分布）计算持有期为1天、置信水平为99%的风险价值", round(VaR99_1day_MCst, 2))

VaR95_10day_MCst = np.sqrt(D2) * VaR95_1day_MCst # 持有期为10天、置信水平为95%的风险价值
VaR99_10day_MCst = np.sqrt(D2) * VaR99_1day_MCst # 持有期为10天、置信水平为99%的风险价值

print("蒙特卡罗模拟法（服从学生t分布）计算持有期为10天、置信水平为95%的风险价值", round(VaR95_10day_MCst, 2))
print("蒙特卡罗模拟法（服从学生t分布）计算持有期为10天、置信水平为99%的风险价值", round(VaR99_10day_MCst, 2))

# 第4步：为了进行比较，假定资产收益率服从正态分布，运用蒙特卡罗模拟法计算投资组合的风险价值，为了减少输入而运用for语句
P = np.array(price.iloc[-1]) # 单个资产的最新收盘价或净值（数组格式）

epsilpn_norm = npr.standard_normal(I) # 从正态分布中抽取样本

P_new = np.zeros(shape = (I, len(R_mean))) # 创建存放模拟下一个交易日单一资产价格的初始数组

for i in range(len(R_mean)):
    P_new[:, i] = P[i] * np.exp((R_mean[i] - 0.5 * R_vol[i] ** 2) * dt + R_vol[i] * epsilpn_norm * np.sqrt(dt)) # 依次模拟投资组合每个资产下一个交易日的收盘价

profit_port_norm = (np.dot(P_new / P - 1, W)) * value_port # 投资组合下一个交易日的收益

plt.figure(figsize = (9, 6))
plt.hist(profit_port_norm, bins = 30, facecolor = "y", edgecolor = "k")
plt.xticks(fontsize = 13)
plt.xlabel(u"投资组合模拟的日收益金额", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylabel(u"频数", fontsize = 13)
plt.title(u"通过蒙特卡罗模拟（服从正态分布）得到投资组合日收益金额的直方图", fontsize = 13)
plt.grid()
plt.show()

VaR95_1day_MCnorm = np.abs(np.percentile(a = profit_port_norm, q = (1 - X1) * 100)) # 持有期为1天、置信水平为95%的风险价值
VaR99_1day_MCnorm = np.abs(np.percentile(a = profit_port_norm, q = (1 - X2) * 100)) # 持有期为1天、置信水平为99%的风险价值

print("蒙特卡罗模拟法（服从正态分布）计算持有期为1天、置信水平为95%的风险价值", round(VaR95_1day_MCnorm, 2))
print("蒙特卡罗模拟法（服从正态分布）计算持有期为1天、置信水平为99%的风险价值", round(VaR99_1day_MCnorm, 2))

VaR95_10day_MCnorm = np.sqrt(D2) * VaR95_1day_MCnorm # 持有期为10天、置信水平为95%的风险价值
VaR99_10day_MCnorm = np.sqrt(D2) * VaR99_1day_MCnorm # 持有期为10天、置信水平为99%的风险价值

print("蒙特卡罗模拟法（服从正态分布）计算持有期为10天、置信水平为95%的风险价值", round(VaR95_10day_MCnorm, 2))
print("蒙特卡罗模拟法（服从正态分布）计算持有期为10天、置信水平为99%的风险价值", round(VaR99_10day_MCnorm, 2))



# 15.5 回溯检验、压力测试与压力风险价值
# 沿用前例的投资组合信息，针对运用方差-协方差法计算得到的持有期为1天、置信水平为95%的风险价值，结合2018年至2020年每年的日交易数据，运用回溯检验判断风险价值的合理性
# 第1步：根据之前历史模拟法计算得出的2018年至2020年期间投资组合日收益余额数据，依次生成每一年投资组合日收益金额的时间序列，并且将每年的投资组合日收益与风险价值所对应的亏损进行可视化
profit_2018 = profit_past.loc["2018-01-01" : "2018-12-31"] # 生成2018年投资组合的日收益
profit_2019 = profit_past.loc["2019-01-01" : "2019-12-31"] # 生成2019年投资组合的日收益
profit_2020 = profit_past.loc["2020-01-01" : "2020-12-31"] # 生成2020年投资组合的日收益

VaR_2018_neg = -VaR95_1day_VCM * np.ones_like(profit_2018) # 创建2018年风险价值对应亏损的数组
VaR_2019_neg = -VaR95_1day_VCM * np.ones_like(profit_2019) # 创建2019年风险价值对应亏损的数组
VaR_2020_neg = -VaR95_1day_VCM * np.ones_like(profit_2020) # 创建2020年风险价值对应亏损的数组

VaR_2018_neg = pd.DataFrame(data = VaR_2018_neg, index = profit_2018.index) # 创建2018年风险价值对应亏损的时间序列
VaR_2019_neg = pd.DataFrame(data = VaR_2019_neg, index = profit_2019.index) # 创建2019年风险价值对应亏损的时间序列
VaR_2020_neg = pd.DataFrame(data = VaR_2020_neg, index = profit_2020.index) # 创建2020年风险价值对应亏损的时间序列

plt.figure(figsize = (9, 12))
plt.subplot(3, 1, 1)
plt.plot(profit_2018, "b-", label = u"2018年投资组合日收益")
plt.plot(VaR_2018_neg, "r-", label = u"风险价值对应的亏损", lw = 2.0)
plt.ylabel(u"收益")
plt.legend(fontsize = 12)
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(profit_2019, "b-", label = u"2019年投资组合日收益")
plt.plot(VaR_2019_neg, "r-", label = u"风险价值对应的亏损", lw = 2.0)
plt.ylabel(u"收益")
plt.legend(fontsize = 12)
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(profit_2020, "b-", label = u"2020年投资组合日收益")
plt.plot(VaR_2020_neg, "r-", label = u"风险价值对应的亏损", lw = 2.0)
plt.ylabel(u"收益")
plt.legend(fontsize = 12)
plt.grid()
plt.show()
# 当投资组合日亏损触及风险价值对应的亏损这条直线时，就表明亏损大于风险价值

# 第2步：计算在2018年至2020年期间，每年的交易天数、每年内投资组合日亏损超出风险价值对应亏损的具体天数以及占当年交易天数的比重
days_2018 = len(profit_2018) # 2018年的全部交易天数
days_2019 = len(profit_2019) # 2019年的全部交易天数
days_2020 = len(profit_2020) # 2020年的全部交易天数
print("2018年的全部交易天数", days_2018)
print("2019年的全部交易天数", days_2019)
print("2020年的全部交易天数", days_2020)

dayexcept_2018 = len(profit_2018[profit_2018["投资组合的模拟日收益"] < -VaR95_1day_VCM]) # 用2018年数据进行回测检验并且计算超过风险价值对应亏损的天数
dayexcept_2019 = len(profit_2019[profit_2019["投资组合的模拟日收益"] < -VaR95_1day_VCM]) # 用2019年数据进行回测检验并且计算超过风险价值对应亏损的天数
dayexcept_2020 = len(profit_2020[profit_2020["投资组合的模拟日收益"] < -VaR95_1day_VCM]) # 用2020年数据进行回测检验并且计算超过风险价值对应亏损的天数
print("2018年超过风险价值对应亏损的天数", dayexcept_2018)
print("2019年超过风险价值对应亏损的天数", dayexcept_2019)
print("2020年超过风险价值对应亏损的天数", dayexcept_2020)

ratio_2018 = dayexcept_2018 / days_2018 # 2018年超过风险价值对应亏损的天数占全年交易天数的比例
ratio_2019 = dayexcept_2019 / days_2019 # 2019年超过风险价值对应亏损的天数占全年交易天数的比例
ratio_2020 = dayexcept_2020 / days_2020 # 2020年超过风险价值对应亏损的天数占全年交易天数的比例
print("2018年超过风险价值对应亏损的天数占全年交易天数的比例", round(ratio_2018, 4))
print("2019年超过风险价值对应亏损的天数占全年交易天数的比例", round(ratio_2019, 4))
print("2020年超过风险价值对应亏损的天数占全年交易天数的比例", round(ratio_2020, 4))
# 该年超过风险价值对应亏损的天数占全年交易天数的比例大于5%则原来方法（方差-协方差法）计算风险价值的模型对该年不适用



# 15.5.2 压力测试
# 沿用前例的投资组合信息，计算该投资组合的压力风险价值
# 计算持有期为10天、置信水平分别为95%和99%的投资组合压力风险价值
# 第1步：计算压力期间投资组合的日收益时间序列并且进行可视化
# TODO: 导入5个股票的数据
price_stress = pd.read_excel("D:/Python/投资组合配置资产压力期间的每日价格.xlsx", sheet_name = "Sheet1", header = 0,
                            index_col = 0) # 导入外部资产
price_stress = price_stress.dropna() # 删除缺失值
price_stress.index = pd.DatetimeIndex(price_stress.index) # 将数据框行索引转换为datetime格式

R_stress = np.log(price_stress / price_stress.shift(1)) # 计算对数收益率
R_stress = R_stress.dropna() # 删除缺失值

profit_stress = np.dot(R_stress, value_past) # 压力期间投资组合的日收益金额（变量value_past在前例已设定）
profit_stress = pd.DataFrame(data = profit_stress, index = R_stress.index, columns = ["投资组合的模拟日收益"]) # 转换为数据框
profit_stress = describe() # 查看描述性统计指标

profit_zero = np.zeros_like(profit_stress) # 创建压力期间收益为0的数组
profit_zero = pd.DataFrame(data = profit_zero, index = profit_stress.index) # 转换为数据框

plt.figure(figsize = (9, 6))
plt.plot(profit_stress, "b-", label = u"压力期间投资组合的日收益")
plt.plot(profit_zero, "r-", label = u"收益等于0", lw = 2.5)
plt.xlabel(u"日期", fontsize = 13)
plt.xticks(fontsize = 13)
plt.ylabel(u"收益", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"压力期间投资组合的收益表现情况", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()

# 第2步：根据压力期间投资组合的日收益时间序列，计算压力风险价值
SVaR95_1day = np.abs(np.percentile(a = profit_stress, q = (1 - X1) * 100)) # 持有期为1天、置信水平为95%的压力风险价值
SVaR99_1day = np.abs(np.percentile(a = profit_stress, q = (1 - X2) * 100)) # 持有期为1天、置信水平为99%的压力风险价值

print("持有期为1天、置信水平为95%的压力风险价值", round(SVaR95_1day, 2))
print("持有期为1天、置信水平为99%的压力风险价值", round(SVaR99_1day, 2))

SVaR95_10day = np.sqrt(D2) * SVaR95_1day # 持有期为10天、置信水平为95%的压力风险价值
SVaR99_10day = np.sqrt(D2) * SVaR99_1day # 持有期为10天、置信水平为99%的压力风险价值

print("持有期为10天、置信水平为95%的压力风险价值", round(SVaR95_10day, 2))
print("持有期为10天、置信水平为99%的压力风险价值", round(SVaR99_10day, 2))



# 15.6 信用风险价值
# 15.6.4 测度信用风险价值
# 通过Python自定义一个计算投资组合信用风险价值的函数
def CVaR(T, X, L, R, Lambda, rou):
    """
    定义一个计算投资组合信用风险价值的函数
    T：代表信用风险价值的持有期，单位是年
    X：代表信用风险价值的置信水平
    L：代表投资组合的总金额
    R：代表投资组合中每个主体的违约回收率并且每个主体均相同
    Lambda：代表投资组合中每个主体连续复利的年化违约概率并且每个主体均相同
    rou：代表投资组合中任意两个主体之间的违约相关系数并且均相同
    """
    from scipy.stats import norm # 导入SciPy的子模块stats的函数norm
    from numpy import exp # 导入NumPy模块的函数exp
    C = 1 - exp(-Lambda * T) # 计算每个主体的累积违约概率
    V = norm.cdf((norm.ppf(C) + pow(rou, 0.5) * norm.ppf(X)) / pow(1-rou, 0.5)) # 计算阈值V(T, X)
    VaR = L * (1 - R) * V # 计算信用风险价值
    return VaR

# 假定一家商业银行持有金额为2000亿元的信贷资产组合，该组合共涉及1000笔信贷资产并对应1000家借款主体，
# 经过计算以后发现每家借款主体的连续复利年化违约概率均为1.5%，违约回收率均为50%，违约相关系数等于0.2，
# 求持有期为1年，置信水平为99.9%该组合的信用风险价值
"""
此外，为了考察置信水平、违约概率和违约相关系数这3个重要变量对信用风险价值的影响，需要依次完成3项敏感性分析工作
1.当置信水平取[80%, 99.9%]区间的等差数列并且在其他变量保持不变的情况下，计算相应的信用风险价值
2.当违约概率取[0.5%, 5%]区间的等差数列并且在其他变量保持不变的情况下，计算相应的信用风险价值
3.当违约相关系数取[0.1, 0.6]区间的等差数列并且在其他变量保持不变的情况下，计算相应的信用风险价值
"""
# 第1步：输入相关参数并运用自定义函数CVaR，计算持有期为1年、置信水平为99.9%的信贷资产组合信用风险价值
tenor = 1 # 信用风险价值的持有期
prob = 0.999 # 信用风险价值的置信水平
par = 2e11 # 投资组合的总金额
recovery = 0.5 # 每个借款主体的违约回收率
PD = 0.015 # 每个借款主体的违约概率
corr = 0.2 # 任意两个借款主体之间的违约相关系数

credit_VaR = CVaR(T = tenor, X = prob, L = par, R = recovery, Lambda = PD, rou = corr) # 计算信用风险价值
print("持有期为1年、置信水平为99.9%的信贷资产组合信用风险价值（亿元）", round(credit_VaR / 1e8, 4))
print("信用风险价值占整个信贷资产组合总金额的比重", round(credit_VaR / par, 6))
# 可得信贷资产组合信用风险价值为188.23亿元，且占整个信贷资产组合总金额的比重达到9.41%

# 第2步：置信水平取[80%, 99.9%]区间的等差数列时，计算相应的信用风险价值，并且将置信水平与信用风险价值的关系进行可视化
prob_list = np.linspace(0.8, 0.999, 200) # 创建置信水平的等差数列

CVaR_list1 = CVaR(T = tenor, X = prob_list, L = par, R = recovery, Lambda = PD, rou = corr) # 计算不同置信水平的信用风险价值

plt.figure(figsize = (9, 6))
plt.plot(prob_list, CVaR_list1, "r-", lw = 2.5)
plt.xlabel(u"置信水平", fontsize = 13)
plt.ylabel(u"信用风险价值", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"置信水平与信用风险价值的关系图", fontsize = 13)
plt.grid()
plt.show()

# 第3步：违约概率取[0.5%, 5%]区间的等差数列时，计算相应的信用风险价值，并且将违约概率与信用风险价值的关系进行可视化
PD_list = np.linspace(0.005, 0.05, 200) # 创建违约概率的等差数列

CVaR_list2 = CVaR(T = tenor, X = prob, L = par, R = recovery, Lambda = PD_list, rou = corr) # 计算不同违约概率的信用风险价值

plt.figure(figsize = (9, 6))
plt.plot(PD_list, CVaR_list2, "m-", lw = 2.5)
plt.xlabel(u"违约概率", fontsize = 13)
plt.ylabel(u"信用风险价值", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"违约概率与信用风险价值的关系图", fontsize = 13)
plt.grid()
plt.show()

# 第4步：当违约相关系数取[0.1, 0.6]区间的等差数列时，计算相应的信用风险价值，并且进行可视化
corr_list = np.linspace(0.1, 0.6, 200) # 创建违约相关系数的等差数列

CVaR_list3 = CVaR(T = tenor, X = prob, L = par, R = recovery, Lambda = PD, rou = corr_list) # 计算不同违约相关系数的信用风险价值

plt.figure(figsize = (9, 6))
plt.plot(PD_list, CVaR_list3, "m-", lw = 2.5)
plt.xlabel(u"违约相关系数", fontsize = 13)
plt.ylabel(u"信用风险价值", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"违约相关系数与信用风险价值的关系图", fontsize = 13)
plt.grid()
plt.show()
