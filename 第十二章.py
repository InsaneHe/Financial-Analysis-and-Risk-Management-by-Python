# 第十二章
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl

from 第十一章 import option_BSM

mpl.rcParams["font.sans-serif"] = ["FangSong"]
mpl.rcParams["axes.unicode_minus"] = False
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def delta_EurOpt(S, K, sigma, r, T, optype, positype):
    """
    定义一个欧式期权Delta的函数
    S：代表期权基础资产的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    optype：代表期权的类型，输入optype="call"表示看涨期权，输入其他则表示看跌期权
    positype：代表期权头寸的方向，输入positype="long"表示期权多头，输入其他则表示期权空头
    """
    from scipy.stats import norm  # 从SciPy的子模块stats中导入norm函数
    from numpy import log, sqrt  # 从NumPy模块中导入log、sqrt函数
    d1 = (log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))  # d1的表达式
    if optype == "call":  # 当期权是看涨期权
        if positype == "long":  # 当期权头寸是多头
            delta = norm.cdf(d1)  # 计算期权的delta
        else:  # 当期权头寸是空头
            delta = -norm.cdf(d1)
    else:  # 当期权是看跌期权
        if positype == "long":
            delta = norm.cdf(d1) - 1
        else:
            delta = 1 - norm.cdf(d1)
    return delta


# 假定在2020年7月16日，期权市场上市了以农业银行A股（代码为601288）作为基础资产、行权价格为3.6元/股、期限为6个月的欧式看涨期权和欧式看跌期权，
# 当天农业银行A股收盘价为3.27元/股，以6个月期Shibor作为无风险收益率，并且当天报价是2.377%，股票收益率的年化波动率是19%，
# 分别计算欧式看涨、看跌期权的多头与空头的Delta
S_ABC = 3.27  # 农业银行股价
K_ABC = 3.6  # 期权的行权价格
sigma_ABC = 0.19  # 农业银行股票年化波动率
shibor_6M = 0.02377  # 6个月期Shibor（无风险收益率）
T_ABC = 0.5  # 期权期限

delta_EurOpt1 = delta_EurOpt(S=S_ABC, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC, optype="call",
                             positype="long")  # 计算欧式看涨期权多头的Delta
delta_EurOpt2 = delta_EurOpt(S=S_ABC, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC, optype="call",
                             positype="short")  # 计算欧式看涨期权空头的Delta
delta_EurOpt3 = delta_EurOpt(S=S_ABC, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC, optype="put",
                             positype="long")  # 计算欧式看跌期权多头的Delta
delta_EurOpt4 = delta_EurOpt(S=S_ABC, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC, optype="put",
                             positype="short")  # 计算欧式看跌期权空头的Delta
print("农业银行A股欧式看涨期权多头的Delta", round(delta_EurOpt1, 4))
print("农业银行A股欧式看涨期权空头的Delta", round(delta_EurOpt2, 4))
print("农业银行A股欧式看跌期权多头的Delta", round(delta_EurOpt3, 4))
print("农业银行A股欧式看跌期权空头的Delta", round(delta_EurOpt4, 4))
# 可看出：空头的Delta是多头的相反数

# 沿用上例信息，以欧式看涨期权作为分析对象，对农业银行A股股价（基础资产价格）取值是[2.5, 4.5]区间的等差数列。
# 针对不同股价，依次运用BSM模型计算期权价格并用Delta计算近似期权价格，并进行可视化比较
# 第1步：运用自定义函数option_BSM计算布莱克-斯科尔斯-默顿模型的期权价格
S_list1 = np.linspace(2.5, 4.5, 200)  # 创建农业银行股价的等差数列

value_list = option_BSM(S=S_list1, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC,
                        opt="call")  # 不同基础资产价格对应的期权价格（运用BSM模型）

value_one = option_BSM(S=S_ABC, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC,
                       opt="call")  # 农业银行股价等于3.27元/股（2020年7月16日收盘价）对应的期权价格

value_approx1 = value_one + delta_EurOpt1 * (S_list1 - S_ABC)  # 用Delta计算不同农业银行股价对应的近似期权价格

# 第2步：将运用BSM模型计算得到的期权价格与运用Delta计算得到的近似期权价格进行可视化
plt.figure(figsize=(9, 6))
plt.plot(S_list1, value_list, "b-", label=u"运用BSM模型计算得到的看涨期权价格", lw=2.5)
plt.plot(S_list1, value_approx1, "r-", label=u"运用Delta计算得到的看涨期权近似价格", lw=2.5)
plt.xlabel(u"股票价格", fontsize=13)
plt.ylabel(u"期权价格", fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u"运用BSM模型计算得到的期权价格与运用Delta计算得到的近似期权价格的关系图", fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()
# 可看出：运用Delta计算得到的近似期权价格曲线是运用BSM模型计算得到的期权价格曲线的一条切线，
# 切点为2020年7月16日农业银行股票收盘价3.27元/股，切线斜率为Delta；
# 当基础资产价格围绕着切点变动较小时两种方法计算得到的期权价格较接近；当基础资产价格变动较大时两种方法计算得到的期权价格存在天壤之别

# 12.1.2 基础资产价格、期权期限与期权Delta的关系
# 沿用前例农业银行股票期权的相关信息，同时对农业银行A股股价的取值时[1.0, 6.0]区间的等差数列，其他的参数保持不变，
# 运用Python将基础资产价格（股票价格）与欧式期权多头Delta之间的对应关系可视化
S_list2 = np.linspace(1.0, 6.0, 200)  # 创建农业银行股价的等差数列

Delta_EurCall = delta_EurOpt(S=S_list2, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC,
                             optype="call", positype="long")  # 计算欧式看涨期权的Delta
Delta_EurPut = delta_EurOpt(S=S_list2, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC,
                            optype="put", positype="long")  # 计算欧式看跌期权的Delta

plt.figure(figsize=(9, 6))
plt.plot(S_list2, Delta_EurCall, "b-", label=u"欧式看涨期权多头", lw=2.5)
plt.plot(S_list2, Delta_EurPut, "r-", label=u"欧式看跌期权多头", lw=2.5)
plt.xlabel(u"股票价格", fontsize=13)
plt.ylabel(u"Delta", fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u"股票价格与欧式期权多头Delta", fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()
# 可看出：1.当基础资产价格增大时，期权Delta会增加；2.曲线斜率始终为正，即Gamma大于0；
# 3.当基础资产价格小于行权价格（3.6元/股）时，随着基础资产价格增大，曲线斜率递增；当基础资产价格大于行权价格时，曲线斜率递减；
# 4.当基础资产价格很小（如低于2.5元/股）或很大（如高于5元/股）时，期权Delta会出现饱和现象即曲线变得平坦，此时Delta对基础资产价格变化很不敏感，
# 饱和现象在其他希腊字母上也多次出现

# 沿用前例农业银行A股股票看涨期权的相关信息，对期权期限设定一个取值是在[0.1, 5.0]区间的等差数列，
# 同时将期权分为实值看涨期权、平价看涨期权和虚值看涨期权3类，
# 实值看涨期权对应的股价为4.0元/股，虚值看涨期权对应的股价为3.0元/股，其他参数保持不变
# 运用Python将期权期限与欧式看涨期权多头Delta之间的对应关系可视化
S1 = 4.0  # 实值看涨期权对应的股价
S2 = 3.6  # 平价看涨期权对应的股价
S3 = 3.0  # 虚值看涨期权对应的股价

T_list = np.linspace(0.1, 5.0, 200)  # 创建期权期限的等差数列

Delta_list1 = delta_EurOpt(S1, K_ABC, sigma_ABC, shibor_6M, T_list, "call", "long")  # 实值看涨期权的Delta
Delta_list2 = delta_EurOpt(S2, K_ABC, sigma_ABC, shibor_6M, T_list, "call", "long")  # 平价看涨期权的Delta
Delta_list3 = delta_EurOpt(S3, K_ABC, sigma_ABC, shibor_6M, T_list, "call", "long")  # 虚值看涨期权的Delta

plt.figure(figsize=(9, 6))
plt.plot(T_list, Delta_list1, "b-", label=u"实值看涨期权多头", lw=2.5)
plt.plot(T_list, Delta_list2, "r-", label=u"平价看涨期权多头", lw=2.5)
plt.plot(T_list, Delta_list3, "g-", label=u"虚值看涨期权多头", lw=2.5)
plt.xlabel(u"期权期限（年）", fontsize=13)
plt.ylabel(u"Delta", fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u"期权期限与欧式看涨期权多头Delta的关系图", fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()
# 随着期权期限的增加，实值看涨期权多头的Delta先递减再缓慢递增；
# 平价、虚值看涨期权多头的Delta均是期权期限的递增函数，但是虚值看涨期权多头Delta的增幅要高于平价期权多头Delta

# 12.1.3 基于Delta的对冲
# 沿用前例农业银行股票期权的信息，假定A金融机构在该期权上市首日即2020年7月16日持有该欧式看跌期权多头头寸1000000份，
# 假定1份期权的基础资产是1股股票。同时为了保持期权组合的Delta中性，A金融机构需要在该交易日买入一定数量的农业银行A股股票
# 经过若干交易日，在2020年8月31日，农业银行A股收盘价是3.21元/股，当天6个月期Shibor（连续复利）为2.636%，股票收益率的年化波动率仍为19%。
# 此外，由于期权期限是6个月，因此期权到期日就是2021年1月16日
# 第1步：假定A金融机构明确将采用静态对冲策略，从而在整个对冲期间用于对冲的农业银行股票数量保持不变，
# 计算2020年7月16日需要买入的农业银行A股股票数量以及在2020年8月31日该策略的对冲效果
N_put = 1e6  # 持有看跌期权多头头寸

N_ABC = np.abs(delta_EurOpt3 * N_put)  # 用于对冲的农业银行A股股票数量（变量Delta_EurOpt3在前面已设定）
N_ABC = int(N_ABC)  # 转换为整型
print("2020年7月16日买入基于期权Delta对冲的农业银行A股数量", N_ABC)

import datetime as dt  # 导入datetime模块

T0 = dt.datetime(2020, 7, 16)  # 设置期权初始日（也就是对冲初始日）
T1 = dt.datetime(2020, 8, 31)  # 设置交易日2020年8月31日
T2 = dt.datetime(2021, 1, 16)  # 设置期权到期日
T_new = (T2 - T1).days / 365  # 2020年8月31日至期权到期日的剩余期限（年）

S_Aug31 = 3.21  # 2020年8月31日农业银行A股股价
shibor_Aug31 = 0.02636  # 2020年8月31日6个月期Shibor

put_Jul16 = option_BSM(S=S_ABC, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC,
                       opt="put")  # 期权初始日看跌期权价格
put_Aug31 = option_BSM(S=S_Aug31, K=K_ABC, sigma=sigma_ABC, r=shibor_Aug31, T=T_new,
                       opt="put")  # 2020年8月31日看跌期权价格
print("2020年7月16日农业银行A股欧式看跌期权价格", round(put_Jul16, 4))
print("2020年8月31日农业银行A股欧式看跌期权价格", round(put_Aug31, 4))

port_chagvalue = N_ABC * (S_Aug31 - S_ABC) + N_put * (put_Aug31 - put_Jul16)  # 静态对冲策略下2020年8月31日投资组合的累积盈亏
print("静态对冲策略下2020年8月31日投资组合的累积盈亏", round(port_chagvalue, 2))

# 第2步：计算在2020年8月31日看跌期权的Delta以及保持该交易日期权Delta中性而需要针对基础资产（农业银行A股）新增交易情况
delta_Aug31 = delta_EurOpt(S=S_Aug31, K=K_ABC, sigma=sigma_ABC, r=shibor_Aug31, T=T_new,
                           optype="put", positype="long")  # 计算2020年8月31日的期权Delta
print("2020年8月31日农业银行A股欧式看跌期权Delta", round(delta_Aug31, 4))

N_ABC_new = np.abs(delta_Aug31 * N_put)  # 2020年8月31日保持Delta中性而用于对冲的农业银行A股股票数量
N_ABC_new = int(N_ABC_new)  # 转换为整型
print("2020年8月31日保持Delta中性而用于对冲的农业银行A股股票数量", N_ABC_new)

N_ABC_change = N_ABC_new - N_ABC  # 保持Delta中性而发生的股票数量变化
print("2020年8月31日保持Delta中性而发生的股票数量变化", N_ABC_change)


# 12.1.4 美式期权的Delta
# 运用Python自定义一个计算美式期权Delta的函数，并且按照看涨期权、看跌期权分别进行定义
def delta_AmerCall(S, K, sigma, r, T, N, positype):
    """
    定义一个运用N步二叉树模型计算美式看涨期权Delta的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    positype：代表期权头寸方向，输入positype="long"表示期权多头，输入其他则表示期权空头
    """
    t = T / N  # 计算每一步步长期限（年）
    u = np.exp(sigma * np.sqrt(t))  # 计算基础资产价格上涨时的比例
    d = 1 / u  # 计算基础资产价格下跌时的比例
    p = (np.exp(r * t) - d) / (u - d)  # 计算基础资产价格上涨的概率
    call_matrix = np.zeros((N + 1, N + 1))  # 创建N+1行、N+1列的零矩阵，用于后续存放每个节点的期权价值
    N_list = np.arange(0, N + 1)  # 创建从0到N的自然数数列（数组格式）
    S_end = S * pow(u, N - N_list) * pow(d, N_list)  # 计算期权到期时节点的基础资产价格（按照节点从上到下排序）
    call_matrix[:, -1] = np.maximum(S_end - K, 0)  # 计算期权到期时节点的看涨期权价值（按照节点从上到下排序）
    i_list = list(range(0, N))  # 创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()  # 将列表的元素从大到小重新排序（从N-1到0）
    for i in i_list:
        j_list = np.arange(i + 1)  # 创建从0到i的自然数数列（数组格式）
        Si = S * pow(u, i - j_list) * pow(d, j_list)  # 计算在iΔt时刻各节点上的基础资产价格（按照节点从上到下排序）
        call_strike = np.maximum(Si - K, 0)  # 计算提前行权时的期权收益
        call_nostrike = np.exp(-r * t) * (p * call_matrix[: i + 1, i + 1] +
                                          (1 - p) * call_matrix[1: i + 2, i + 1])  # 计算不提前行权时的期权价值
        call_matrix[: i + 1, i] = np.maximum(call_strike,
                                             call_nostrike)  # 取提前行权时的期权收益与不提前行权时的期权价值中的最大值
    Delta = (call_matrix[0, 1] - call_matrix[1, 1]) / (S * u - S * d)  # 计算期权Delta
    if positype == "long":  # 当期权头寸是多头时
        result = Delta
    else:  # 当期权头寸是空头时
        result = -Delta
    return result


def delta_AmerPut(S, K, sigma, r, T, N, positype):
    """
    定义一个运用N步二叉树模型计算美式看跌期权Delta的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    positype：代表期权头寸方向，输入positype="long"表示期权多头，输入其他则表示期权空头
    """
    t = T / N  # 计算每一步步长期限（年）
    u = np.exp(sigma * np.sqrt(t))  # 计算基础资产价格上涨时的比例
    d = 1 / u  # 计算基础资产价格下跌时的比例
    p = (np.exp(r * t) - d) / (u - d)  # 计算基础资产价格上涨的概率
    put_matrix = np.zeros((N + 1, N + 1))  # 创建N+1行、N+1列的零矩阵，用于后续存放每个节点的期权价值
    N_list = np.arange(0, N + 1)  # 创建从0到N的自然数数列（数组格式）
    S_end = S * pow(u, N - N_list) * pow(d, N_list)  # 计算期权到期时节点的基础资产价格（按照节点从上往下排序）
    put_matrix[:, -1] = np.maximum(K - S_end, 0)  # 计算期权到期时节点的看涨期权价值（按照节点从上往下顺序）
    i_list = list(range(0, N))  # 创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()  # 将列表的元素由大到小重新排序（从N-1到0）
    for i in i_list:
        j_list = np.arange(i + 1)  # 创建从0到i的自然数数列（数组格式）
        Si = S * pow(u, i - j_list) * pow(d, j_list)  # 计算在iΔt时刻各节点上的基础资产价格（按照节点从上往下排序）
        put_strike = np.maximum(K - Si, 0)  # 计算提前行权时的期权收益
        put_nostrike = np.exp(-r * t) * (p * put_matrix[: i + 1, i + 1] + (1 - p) *
                                         put_matrix[1: i + 2, i + 1])  # 计算不提前行权时的期权价值
        put_matrix[: i + 1, i] = np.maximum(put_strike, put_nostrike)  # 取提前行权时的期权收益与不提前行权时的期权价值中的最大值
    Delta = (put_matrix[0, 1] - put_matrix[1, 1]) / (S * u - S * d)  # 计算期权Delta=(Π1,1-Π1,0)/(S0u-S0d)
    if positype == "long":  # 当期权头寸是多头时
        result = Delta
    else:  # 当期权头寸是空头时
        result = -Delta
    return result


step = 100  # 二叉树模型的步数

delta_AmerOpt1 = delta_AmerCall(S=S_ABC, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC, N=step,
                                positype="long")  # 计算美式看涨期权多头的Delta
delta_AmerOpt2 = delta_AmerCall(S=S_ABC, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC, N=step,
                                positype="short")  # 计算美式看涨期权空头的Delta
delta_AmerOpt3 = delta_AmerPut(S=S_ABC, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC, N=step,
                               positype="long")  # 计算美式看跌期权多头的Delta
delta_AmerOpt4 = delta_AmerPut(S=S_ABC, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC, N=step,
                               positype="short")  # 计算美式看跌期权空头的Delta
print("农业银行A股美式看涨期权多头的Delta", round(delta_AmerOpt1, 4))
print("农业银行A股美式看涨期权空头的Delta", round(delta_AmerOpt2, 4))
print("农业银行A股美式看跌期权多头的Delta", round(delta_AmerOpt3, 4))
print("农业银行A股美式看跌期权空头的Delta", round(delta_AmerOpt4, 4))


# 可看出：美式看涨期权的Delta绝对值与欧式看涨期权的Delta绝对值是很接近的；美式看跌期权的Delta绝对值则大于欧式看跌期权的Delta绝对值；
# 因此相比欧式看跌期权，美式看跌期权的价值对基础资产价格更加敏感


# 12.2 期权的Gamma
# 12.2.1 欧式期权的Gamma
# 通过Python自定义一个计算欧式期权的Gamma的函数
def gamma_EurOpt(S, K, sigma, r, T):
    """
    定义一个计算欧式期权Gamma的函数
    S：代表期权基础资产的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的剩余期限（年）
    """
    from numpy import exp, log, pi, sqrt  # 从NumPy模块导入exp、log、pi和sqrt函数
    d1 = (log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))  # 计算d1
    gamma = exp(-pow(d1, 2) / 2) / (S * sigma * sqrt(2 * pi * T))  # 计算Gamma
    return gamma


# 沿用前例农业银行股票期权信息，并运用自定义函数gamma_EurOpt计算该期权的Gamma
gamma_Eur = gamma_EurOpt(S=S_ABC, K=K_ABC, sigma=sigma_ABC, r=shibor_6M, T=T_ABC)  # 计算期权的Gamma
print("农业银行A股欧式期权的Gamma", round(gamma_Eur, 4))

# 沿用前例的相关信息，依然以看涨期权作为分析对象，对农业银行A股股价的取值依然是[2.5, 4.5]区间的等差数列。
# 针对不同的股价，通过公式并运用Python计算期权的近似价格，
# 在前例中已计算出通过BSM模型得到的期权价格以及仅运用Delta计算得到的期权近似价格，通过可视化对比这3中方法计算出的期权价格
value_approx2 = (value_one + delta_EurOpt1 * (S_list1 - S_ABC) +
                 0.5 * gamma_Eur * pow(S_list1 - S_ABC, 2))  # 用Delta和Gamma计算近似的期权价格

plt.figure(figsize=(9, 6))
plt.plot(S_list1, value_list, "b-", label=u"运用BSM模型计算的看涨期权价格", lw=2.5)
plt.plot(S_list1, value_approx1, "r-", label=u"仅用Delta计算的看涨期权近似价格", lw=2.5)
plt.plot(S_list1, value_approx2, "m-", label=u"用Delta和Gamma计算的看涨期权近似价格", lw=2.5)
plt.plot(S_ABC, value_one, "o", label=u"股价等于3.27元/股对应的期权价格", lw=2.5)
plt.xlabel(u"股票价格", fontsize=13)
plt.ylabel(u"期权价格", fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u"运用BSM模型、仅用Delta以及用Delta和Gamma计算的期权价格", fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()
# 用Delta和Gamma计算得到的期权价格与用BSM模型计算得到的期权价格是比较接近的，
# 当基础资产价格出现较大的增量变化时，两种方法计算得到的期权价格也依然比较接近；
# 如果基础资产价格出现较大的减量变化时，用Delta和Gamma计算期权价格的精确度会大打折扣，
# 说明Gamma对于期权价格向上修正的效应会优于向下修正的效应

# 12.2.2 基础资产价格、期权期限与期权Gamma的关系
# 基础资产价格与期权Gamma的关系
# 沿用前例信息，对农业银行A股股票价格（基础资产价格）依然设定一个取值是在[1.0, 6.0]区间的等差数列，其他的参数保持不变，并运用Python将股票价格与期权Gamma之间的对应关系可视化
gamma_list = gamma_EurOpt(S=S_list2, K=K_ABC, sigma=sigma_ABC, r=shibor_6M,
                          T=T_ABC)  # 计算对应不同股票价格的期权Gamma（变量S_list2在前例中已设定）

plt.figure(figsize=(9, 6))
plt.plot(S_list2, gamma_list, "b-", lw=2.5)
plt.xlabel(u"股票价格", fontsize=13)
plt.ylabel("Gamma", fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u"股票价格与期权Gamma的关系图", fontsize=13)
plt.grid()
plt.show()
# 可看出：这条曲线的形态较接近正态分布。当基础资产价格小于行权价格即看涨期权虚值，看跌期权实值时，期权Gamma是基础资产价格的递增函数；
# 反之则期权Gamma是基础资产价格的递减函数；当基础资产价格接近于行权价格即看涨期权接近于平价时，期权Gamma最大；
# 当曲线处于底部，本例即股价低于2元/股以及高于5元/股，期权Gamma出现饱和现象，即对基础资产价格不敏感

# 期权期限与期权Gamma的关系
gamma_list1 = gamma_EurOpt(S=S1, K=K_ABC, sigma=sigma_ABC, r=shibor_6M,
                           T=T_list)  # 实值看涨期权的Gamma（变量S1和T_list在前例中已设定）
gamma_list2 = gamma_EurOpt(S=S2, K=K_ABC, sigma=sigma_ABC, r=shibor_6M,
                           T=T_list)  # 实值看涨期权的Gamma（变量S2在前例中已设定）
gamma_list3 = gamma_EurOpt(S=S3, K=K_ABC, sigma=sigma_ABC, r=shibor_6M,
                           T=T_list)  # 实值看涨期权的Gamma（变量S3在前例中已设定）

plt.figure(figsize=(9, 6))
plt.plot(T_list, gamma_list1, "b-", label=u"实值看涨期权", lw=2.5)
plt.plot(T_list, gamma_list2, "r-", label=u"平价看涨期权", lw=2.5)
plt.plot(T_list, gamma_list3, "g-", label=u"虚值看涨期权", lw=2.5)
plt.xlabel(u"期权期限", fontsize=13)
plt.ylabel("Gamma", fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title(u"期权期限与期权Gamma的关系图", fontsize=13)
plt.legend(fontsize=13)
plt.grid()
plt.show()


# 可看出：对于平价看涨期权，Gamma时期权期限的递减函数；
# 期限越短的平价看涨期权Gamma越高，即越接近合约到期日，平价看涨期权的Delta对于基础资产价格的变动越敏感；
# 无论实值看涨期权还是虚值看涨期权，期权期限较短时，Gamma是期权期限的递增函数，当期权期限拉长时，Gamma是期权期限的递减函数

# 12.2.3 美式期权的Gamma
def gamma_AmerCall(S, K, sigma, r, T, N):
    """
    定义一个运用N步二叉树模型计算美式看涨期权Gamma的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    t = T / N  # 计算每一步步长的期限（年）
    u = np.exp(sigma * np.sqrt(t))  # 计算基础资产价格上涨时的比例
    d = 1 / u  # 计算基础资产价格下跌时的比例
    p = (np.exp(r * t) - d) / (u - d)  # 计算基础资产价格上涨的概率
    call_matrix = np.zeros((N + 1, N + 1))  # 创建N+1行，N+1列的矩阵并且元素均为0，用于后续存放每个节点的期权价值
    N_list = np.arange(0, N + 1)  # 创建从0到N的自然数数列（数组格式）
    S_end = S * pow(u, N - N_list) * pow(d, N_list)  # 计算期权到期时节点的基础资产价格（按照节点从上往下排序）
    call_matrix[:, -1] = np.maximum(S_end - K, 0)  # 计算期权到期时节点的看涨期权价值（按照节点从上往下排序）
    i_list = list(range(0, N))  # 创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()  # 将列表的元素由大到小重新排序（从N-1到0）
    for i in i_list:
        j_list = np.arange(i + 1)  # 创建从0到i的自然数数列（数组格式）
        Si = S * pow(u, i - j_list) * pow(d, j_list)  # 计算在iΔt时刻各节点上的基础资产价格（按照节点从上往下排序）
        call_strike = np.maximum(Si - K, 0)  # 计算提前行权时的期权收益
        call_nostrike = np.exp(-r * t) * (p * call_matrix[: i + 1, i + 1] + (1 - p) *
                                          call_matrix[1: i + 2, i + 1])  # 计算不提前行权时的期权价值
        call_matrix[: i + 1, i] = np.maximum(call_strike, call_nostrike)  # 取提前行权时的期权收益与不提前行权时的期权价值中的最大值
    Delta1 = (call_matrix[0, 2] - call_matrix[1, 2]) / (S * pow(u, 2) - S)  # 计算一个Delta
    Delta2 = (call_matrix[1, 2] - call_matrix[2, 2]) / (S - S * pow(d, 2))  # 计算另一个Delta
    Gamma = 2 * (Delta1 - Delta2) / (S * pow(u, 2) - S * pow(d, 2))  # 计算美式看涨期权Gamma
    return Gamma


def gamma_AmerPut(S, K, sigma, r, T, N):
    """定义一个运用N步二叉树模型计算美式看跌期权Gamma的函数
    S：代表基础资产当前的价格。
    K：代表期权的行权价格。
    sigma：代表基础资产收益率的波动率（年化）。
    r：代表连续复利的无风险收益率。
    T：代表期权的期限（年）。
    N：代表二叉树模型的步数"""
    t = T / N
    u = np.exp(sigma * np.sqrt(t))
    d = 1 / u
    p = (np.exp(r * t) - d) / (u - d)
    put_matrix = np.zeros((N + 1, N + 1))
    N_list = np.arange(0, N + 1)
    S_end = S * pow(u, N - N_list) * pow(d, N_list)
    put_matrix[:, -1] = np.maximum(K - S_end, 0)
    i_list = list(range(0, N))
    i_list.reverse()
    for i in i_list:
        j_list = np.arange(i + 1)
        Si = S * pow(u, i - j_list) * pow(d, j_list)
        put_strike = np.maximum(K - Si, 0)
        put_nostrike = np.exp(-r * t) * (p * put_matrix[: i + 1, i + 1] + (1 - p) * put_matrix[1: i + 2, i + 1])
        put_matrix[: i + 1, i] = np.maximum(put_strike, put_nostrike)
    Delta1 = (put_matrix[0, 2] - put_matrix[1, 2]) / (S * pow(u, 2) - S)
    Delta2 = (put_matrix[1, 2] - put_matrix[2, 2]) / (S - S * pow(d, 2))
    Gamma = 2 * (Delta1 - Delta2) / (S * pow(u, 2) - S * pow(d, 2))
    return Gamma

# 沿用前例的相关信息，同时将期权类型调整为美式看涨期权、美式看跌期权，运用自定义函数gamma_AmerCall和gamma_AmerPut直接计算期权的Gamma，并且设定二叉树模型的步数依然是100步
gamma_AmerOpt1 = gamma_AmerCall(S = S_ABC, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC,
                                N = step) # 计算美式看涨期权的Gamma（变量step在前例中已设定）
gamma_AmerOpt2 = gamma_AmerPut(S = S_ABC, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC,
                                N = step) # 计算美式看跌期权的Gamma
print("农业银行A股美式看涨期权的Gamma", round(gamma_AmerOpt1, 4))
print("农业银行A股美式看跌期权的Gamma", round(gamma_AmerOpt2, 4))
# 可看出：美式看涨期权的Gamma与欧式看涨期权的Gamma较接近；美式看跌期权的Gamma显著高于欧式看跌期权的Gamma



# 12.3 期权的Theta
# 12.3.1 欧式期权的Theta

