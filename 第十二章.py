# 第十二章
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl

from 第十一章 import option_BSM, American_put, American_call

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
# 通过Python自定义一个计算欧式期权Theta的函数
def theta_EurOpt(S, K, sigma, r, T, optype):
    """
    定义一个计算欧式期权Theta的函数
    S：代表基础资产的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的剩余期限（年）
    optype：代表期权的类型，输入optype="call"表示看涨期权，输入其他则表示看跌期权
    """
    from numpy import exp, log, pi, sqrt  # 从NumPy模块导入exp、log、pi和sqrt函数
    from scipy.stats import norm  # 从SciPy的子模块stats中导入norm函数
    d1 = (log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))  # 计算参数d1
    d2 = d1 - sigma * sqrt(T)  # 计算参数d2
    theta_call = (-(S * sigma * exp(-pow(d1, 2) / 2)) / (2 * sqrt(2 * pi * T)) - r * K * exp(-r * T) *
                  norm.cdf(d2))  # 计算看涨期权的Theta
    theta_put = theta_call + r * K * np.exp(-r * T)  # 计算看跌期权的Theta
    if optype == "call":  # 当期权是看涨期权时
        theta = theta_call
    else:  # 当期权是看跌期权时
        theta = theta_put
    return theta


# 沿用前例的信息，运用自定义函数theta_EurOpt，分别计算农业银行A股欧式看涨、看跌期权的Theta
day1 = 365  # 1年的日历天数
day2 = 252  # 1年的交易天数

theta_EurCall = theta_EurOpt(S = S_ABC, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC,
                             optype = "call")  # 计算欧式看涨期权的Theta
theta_EurPut = theta_EurOpt(S = S_ABC, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC,
                            optype = "put")  # 计算欧式看跌期权的Theta

print("农业银行A股欧式看涨期权Theta", round(theta_EurCall, 6))
print("农业银行A股欧式看涨期权每日历天Theta", round(theta_EurCall / day1, 6))
print("农业银行A股欧式看涨期权每交易日Theta", round(theta_EurCall / day2, 6))
print("农业银行A股欧式看跌期权Theta", round(theta_EurPut, 6))
print("农业银行A股欧式看跌期权每日历天Theta", round(theta_EurPut / day1, 6))
print("农业银行A股欧式看跌期权每交易日Theta", round(theta_EurPut / day2, 6))

# 12.3.2 基础资产价格、期权期限与期权Theta的关系
# 沿用前例信息，对农业银行A股股价取值依然是[1.0, 6.0]区间的等差数列，其他参数保持不变，运用Python将基础资产价格（股票价格）与期权Theta之间的对应关系可视化
theta_EurCall_list = theta_EurOpt(S = S_list2, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC,
                                  optype = "call")  # 计算针对不同股价的欧式看涨期权Theta
theta_EurPut_list = theta_EurOpt(S = S_list2, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC,
                                 optype = "put")  # 计算针对不同股价的欧式看跌期权Theta

plt.figure(figsize = (9, 6))
plt.plot(S_list2, theta_EurCall_list, "b-", label = u"欧式看涨期权", lw = 2.5)
plt.plot(S_list2, theta_EurPut_list, "r-", label = u"欧式看跌期权", lw = 2.5)
plt.xlabel(u"股票价格", fontsize = 13)
plt.ylabel("Theta", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"股票价格与期权Theta的关系图", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 1.无论是看涨期权还是看跌期权，期权Theta与基础资产价格之间关系的曲线形状很相似；
# 2.当基础资产价格等于行权价格即平价期权时，无论看涨期权还是看跌期权，Theta是负值并且绝对值最大，即期权价格对时间的变化非常敏感；
# 3.当基础资产价格大于行权价格时，Theta的绝对值递减，看涨期权的Theta趋于某个负值，看跌期权的Theta趋于0；
# 4.当基础资产价格小于行权价格时，对看跌期权，基础资产价格下降且小于3元/股时Theta有负转正并趋于一个正值，而看涨期权的Theta趋于0；
# 5.当基础资产价格很低（如小于2.5元/股）或很高（如大于5.5元/股），期权Theta就会饱和

# 沿用前例信息，对看涨期权的期限取值在[0.1, 5.0]区间的等差数列，同时依然将看涨期权分为实值看涨期权（对应股价为4.0元/股）、
# 平价看涨期权和虚值看涨期权（对应股价为3.0元/股），其他参数也保持不变，运用Python将期权期限与看涨期权Theta之间的对应关系可视化
theta_list1 = theta_EurOpt(S = S1, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_list,
                           optype = "call")  # 实值看涨期权的Theta
theta_list2 = theta_EurOpt(S = S2, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_list,
                           optype = "call")  # 平价看涨期权的Theta
theta_list3 = theta_EurOpt(S = S3, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_list,
                           optype = "call")  # 虚值看涨期权的Theta

plt.figure(figsize = (9, 6))
plt.plot(T_list, theta_list1, "b-", label = u"实值看涨期权", lw = 2.5)
plt.plot(T_list, theta_list2, "r-", label = u"平价看涨期权", lw = 2.5)
plt.plot(T_list, theta_list3, "g-", label = u"虚值看涨期权", lw = 2.5)
plt.xlabel(u"期权期限", fontsize = 13)
plt.ylabel("Theta", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"期权期限与期权Theta的关系图", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()

# 12.3.3 美式期权的Theta
def theta_AmerCall(S, K, sigma, r, T, N):
    """
    定义一个运用N步二叉树模型计算没事看涨期权Theta的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    t = T / N  # 计算每一步步长期限（年）
    u = np.exp(sigma * np.sqrt(t))  # 计算基础资产价格上涨时的比例
    d = 1 / u  # 计算基础资产价格下跌时的比例
    p = (np.exp(r * t) - d) / (u - d)  # 计算基础资产价格上涨时的比例
    call_matrix = np.zeros((N+1, N+1))  # 创建N+1行，N+1列的零矩阵，用于后续存放每个节点的期权价值
    N_list = np.arange(0, N+1)  # 创建从0到N的自然数数列（数组格式）
    S_end = S * pow(u, N-N_list) * pow(d, N_list)  # 计算期权到期时节点的基础资产价格（按照节点从上到下排序）
    call_matrix[:, -1] = np.maximum(S_end - K, 0)  # 计算期权到期时节点的看涨期权价值（按照节点从上到下排序）
    i_list = list(range(0, N))  # 创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()  # 将列表的元素由大到小重新排序（从N-1到0）
    for i in i_list:
        j_list = np.arange(i+1)  # 创建从0到i的自然数数列（数组格式）
        Si = S * pow(u, i-j_list) * pow(d, j_list)  # 计算在iΔt时刻各节点上的基础资产价格（按照节点从上到下排序）
        call_strike = np.maximum(Si - K, 0)  # 计算提前行权时的期权收益
        call_nostrike = np.exp(-r * t) * (p * call_matrix[: i+1, i+1] +
                                          (1 - p) * call_matrix[1: i+2, i+1])  # 计算不提前行权时的期权价值
        call_matrix[: i+1, i] = np.maximum(call_strike, call_nostrike)
    Theta = (call_matrix[1, 2] - call_matrix[0, 0]) / (2 * t)
    return Theta

def theta_AmerPut(S,K,sigma,r,T,N):
    """
    定义一个运用N步二叉树模型计算美式看跌期权Theta的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    t = T / N
    u = np.exp(sigma * np.sqrt(t))
    d = 1 / u
    p = (np.exp(r * t) - d) / (u - d)
    put_matrix = np.zeros((N+1, N+1))
    N_list = np.arange(0, N+1)
    S_end = S * pow(u, N-N_list) * pow(d, N_list)
    put_matrix[:, -1] = np.maximum(K - S_end, 0)
    i_list = list(range(0, N))
    i_list.reverse()
    for i in i_list:
        j_list = np.arange(i+1)
        Si = S * pow(u, i-j_list) * pow(d, j_list)
        put_strike = np.maximum(K - Si, 0)
        put_nostrike = np.exp(-r * t) * (p * put_matrix[: i+1, i+1] + (1 - p) * put_matrix[1: i+2, i+1])
        put_matrix[: i+1, i] = np.maximum(put_strike, put_nostrike)
    Theta = (put_matrix[1, 2] - put_matrix[0, 0]) / (2 * t)
    return Theta

# 沿用前例信息，同时将该期权调整为美式看涨期权、美式看跌期权，运用自定义函数theta_AmerCall和theta_AmerPut计算美式期权的Theta，
# 并且二叉树模型的步数依然是100步
theta_AmerOpt1 = theta_AmerCall(S = S_ABC, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC, N = step)
theta_AmerOpt2 = theta_AmerPut(S = S_ABC, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC, N = step)
print("农业银行A股美式看涨期权的Theta", round(theta_AmerOpt1, 4))
print("农业银行A股美式看跌期权的Theta", round(theta_AmerOpt2, 4))
# 可看出：美式看涨期权的Theta与欧式看涨期权的Theta很接近，但是欧式看跌期权的Theta从绝对值来看明显低于美式看跌期权的Theta，
# 即美式看跌期权价值相比欧式看跌期权价值对时间的流逝更敏感



# 12.4 期权的Vega
# 12.4.1 欧式期权的Vega
# 通过Python自定义一个计算欧式期权Vega的函数
def vega_EurOpt(S, K, sigma, r, T):
    """
    定义一个计算欧式期权Vega的函数
    S：代表基础资产的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的剩余期限（年）
    """
    from numpy import exp, log, pi, sqrt  # 从NumPy模块导入exp、log、pi以及sqrt函数
    d1 = (log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))  # 计算参数d1
    vega = S * sqrt(T) * exp(-pow(d1, 2) / 2) / sqrt(2 * pi)  # 计算期权的Vega
    return vega

# 沿用前例的信息，并运用自定义函数vega_EurOpt，计算农业银行A股欧式期权的Vega以及当波动率增加1%时期权价格的变动额
vega_Eur = vega_EurOpt(S = S_ABC, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC)  # 计算欧式期权的Vega
print("农业银行A股欧式期权的Vega", round(vega_Eur, 4))

sigma_chg = 0.01

value_chg = vega_Eur * sigma_chg  # 波动率增加1%导致期权价格变动额
print("波动率增加1%导致期权价格变动额", round(value_chg, 4))

# 12.4.2 基础资产价格、期权期限与期权Vega的关系
# 沿用前例信息，即对农业银行A股股价取值仍是[1.0, 6.0]区间的等差数列，其他参数保持不变，
# 运用Python将期权的基础资产价格（股票价格）与期权Vega之间的对应关系可视化
vega_list = vega_EurOpt(S = S_list2, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC)

plt.figure(figsize = (9, 6))
plt.plot(S_list2, vega_list, "b-", lw = 2.5)
plt.xlabel(u"股票价格", fontsize = 13)
plt.ylabel("Vega", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"基础资产（股票）价格与期权Vega的关系图", fontsize = 13)
plt.grid()
plt.show()
# 可看出：基础资产价格与期权Vega之间的关系曲线形态类似正态分布，类似于基础资产价格与期权Gamma的关系图；
# 当基础资产价格等于行权价格时，期权的Vega达到最大；当基础资产价格小于行权价格时，期权的Vega是基础资产价格的增函数；
# 当基础资产价格大于行权价格时，期权的Vega是基础资产价格的减函数；
# 当基础资产价格很小（如小于2.0元/股）或很大（如大于5.5元/股），期权Vega会出现饱和现象，对基础资产价格变化就很不敏感

# 沿用前例信息，即对看涨期权的期限取值在[0.1, 5.0]区间的等差数列，同时还是将看涨期权分为实值看涨期权（对应股价为4.0元/股）、
# 平价看涨期权和虚值看涨期权（对应股价为3.0元/股），其他参数也保持不变
# 运用Python将期权期限与期权Vega之间的对应关系可视化
vega_list1 = vega_EurOpt(S = S1, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_list)
vega_list2 = vega_EurOpt(S = S2, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_list)
vega_list3 = vega_EurOpt(S = S3, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_list)

plt.figure(figsize = (9, 6))
plt.plot(T_list, vega_list1, "b-", label = u"实值看涨期权", lw = 2.5)
plt.plot(T_list, vega_list2, "r-", label = u"平价看涨期权", lw = 2.5)
plt.plot(T_list, vega_list3, "g-", label = u"虚值看涨期权", lw = 2.5)
plt.xlabel(u"期权期限", fontsize = 13)
plt.ylabel("Vega", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"期权期限与期权Vega的关系图", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 可看出：无论是平价、实值还是虚值看涨期权，Vega都是期权期限的递增函数；当波动率发生变化时，期限长的期权要比期限较短的期权在价格上变化更大。
# 在相同期限的条件下，平价看涨期权的Vega高于实值看涨期权，实值看涨期权的Vega又高于虚值看涨期权；
# 这种排序关系并非恒定不变，其会随着期权实值和虚值程度的变化而发生改变

# 12.4.3 美式期权的Vega
# 运用Python自定义一个计算美式期权Vega的函数，并且按照看涨期权、看跌期权分别设定
def vega_AmerCall(S, K, sigma, r, T, N):
    """
    定义一个运用N步二叉树模型计算美式看涨期权Vega的函数，并且假定基础资产收益率的波动率是增加0.0001
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    Value1 = American_call(S, K, sigma, r, T, N)  # 原二叉树模型计算的期权价值
    Value2 = American_call(S, K, sigma+0.0001, r, T, N)  # 新二叉树模型计算的期权价值
    vega = (Value2 - Value1) / 0.0001  # 计算美式看涨期权的Vega
    return vega

def vega_AmerPut(S, K, sigma, r, T, N):
    """
    定义一个运用N步二叉树模型计算美式看跌期权Vega的函数，依然假定基础资产收益率的波动率是增加0.0001
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    Value1 = American_put(S, K, sigma, r, T, N)  # 原二叉树模型计算的期权价值
    Value2 = American_put(S, K, sigma + 0.0001, r, T, N)  # 新二叉树模型计算的期权价值
    vega = (Value2 - Value1) / 0.0001  # 计算美式看跌期权的Vega
    return vega

# 沿用前例的信息，同时将期权调整为美式看涨期权、看跌期权，运用自定义函数vega_AmerCall和vega_AmerPut计算美式期权的vega，并且二叉树模型的步数依然是100步
vega_AmerOpt1=vega_AmerCall(S = S_ABC, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC, N = step)
vega_AmerOpt2=vega_AmerPut(S = S_ABC, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC, N = step)
print('农业银行A股美式看涨期权的Vega', round(vega_AmerOpt1, 4))
print('农业银行A股美式看跌期权的Vega', round(vega_AmerOpt2, 4))
# 可看出：美式看涨期权的Vega与欧式看涨期权的Vega比较接近；美式看跌期权的Vega则显著低于欧式期权的Vega即美式看跌期权对波动率的敏感性更低



# 12.5 期权的Rho
# 12.5.1 欧式期权的Rho
# 通过Python自定义一个计算欧式期权Rho的函数
def rho_EurOpt(S, K, sigma, r, T, optype):
    """
    定义一个计算欧式期权Rho的函数
    S：代表基础资产的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的剩余期限（年）
    optype：代表期权的类型，输入optype="call"表示看涨期权，输入其他则表示看跌期权
    """
    from numpy import exp, log, sqrt  # 从NumPy模块导入exp、log和sqrt函数
    from scipy.stats import norm  # 从SciPy的子模块stats中导入norm函数
    d2 = (log(S / K) + (r - pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))  # 计算参数d2
    if optype == "call":  # 当期权是看涨期权时
        rho = K * T * exp(-r * T) * norm.cdf(d2)  # 计算期权的Rho
    else:  # 当期权是看跌期权时
        rho = -K * T * exp(-r * T) * norm.cdf(-d2)
    return rho

# 沿用前例信息，运用自定义函数rho_EurOpt，依次计算农业银行A股欧式看涨期权、看跌期权的Rho，以及当无风险收益率上涨10个基点（0.1%）时期权价格的变动额
rho_EurCall = rho_EurOpt(S = S_ABC, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC,
                         optype = "call")  # 计算看涨期权的Rho
rho_EurPut = rho_EurOpt(S = S_ABC, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC,
                         optype = "put")  # 计算看跌期权的Rho
print("农业银行A股欧式看涨期权的Rho", round(rho_EurCall, 4))
print("农业银行A股欧式看跌期权的Rho", round(rho_EurPut, 4))

r_chg = 0.001  # 无风险收益率的变化

call_chg = rho_EurCall * r_chg  # 无风险收益率变化导致欧式看涨期权价格的变动额
put_chg = rho_EurPut * r_chg  # 无风险收益率变化导致欧式看跌期权价格的变动额
print("无风险收益率上涨10个基点导致欧式看涨期权价格变化", round(call_chg, 4))
print("无风险收益率上涨10个基点导致欧式看跌期权价格变化", round(put_chg, 4))
# 可看出：农业银行A股欧式看涨期权的Rho为正数，欧式看跌期权的Rho为负数；就Rho的绝对值而言，欧式看跌期权明显大于欧式看涨期权即无风险收益率变化对欧式看跌期权的影响更大

# 12.5.2 基础资产价格、期权期限与期权Rho的关系
# 沿用前例信息，对农业银行A股股价取值依然是在[1.0, 6.0]区间的等差数列，其他参数也保持不变，
# 运用Python将期权的基础资产价格（股票价格）与期权Rho之间的对应关系可视化
rho_EurCall_list=rho_EurOpt(S = S_list2, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC,
                            optype = 'call')  # 对应不同股价的看涨期权价格
rho_EurPut_list=rho_EurOpt(S = S_list2, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC,
                           optype = 'put')  # 对应不同股价的看跌期权价格

plt.figure(figsize = (9, 6))
plt.plot(S_list2, rho_EurCall_list, "b-", label = u"欧式看涨期权", lw = 2.5)
plt.plot(S_list2, rho_EurPut_list, "r-", label = u"欧式看跌期权", lw = 2.5)
plt.xlabel(u"股票价格", fontsize = 13)
plt.ylabel("Rho", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"股票价格与期权Rho的关系图", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 可看出：该图类似于基础资产价格与期权Delta的图像；无论看涨期权还是看跌期权，Rho都是基础资产价格的递增函数且实值期权Rho的绝对值大于虚值期权Rho的绝对值；
# 当基础资产价格低于2.5元/股或高于5元/股时，期权Rho有饱和现象，对基础资产价格不再敏感

# 沿用前例信息，看涨期权的期限设定是在[0.1, 5.0]区间的等差数列，同时依然将看涨期权分为实值看涨期权（对应股价为4.0元/股）、
# 平价看涨期权和虚值看涨期权（对应股价为3.0元/股），其他参数不变。运用Python将期权期限与期权Rho之间的对应关系可视化
rho_list1=rho_EurOpt(S = S1, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_list, optype = "call")  # 实值看涨期权的Rho
rho_list2=rho_EurOpt(S = S2, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_list, optype = "call")  # 平价看涨期权的Rho
rho_list3=rho_EurOpt(S = S3, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_list, optype = "call")  # 虚值看涨期权的Rho

plt.figure(figsize = (9, 6))
plt.plot(T_list, rho_list1, "b-", label = u"实值看涨期权", lw = 2.5)
plt.plot(T_list, rho_list2, "r-", label = u"平价看涨期权", lw = 2.5)
plt.plot(T_list, rho_list3, "g-", label = u"虚值看涨期权", lw = 2.5)
plt.xlabel(u"期权期限", fontsize = 13)
plt.ylabel("Rho", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"期权期限与期权Rho的关系图", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 可看出：1.看涨期权Rho都是期权期限的递增函数，越接近到期日，Rho越小，相反则越大；
# 2.相同期限时，实值看涨期权的Rho大于平价看涨期权，平价看涨期权的Rho又大于虚值看涨期权；
# 3.随着期权期限拉长，实值、平价和虚值看涨期权在Rho上的差异会变大

# 12.5.3 美式期权的Rho
# 运用Python自定义一个计算美式期权Rho的函数，并且按照看涨期权、看跌期权分别设定
def rho_AmerCall(S, K, sigma, r, T, N):
    """
    定义一个运用N步二叉树模型计算美式看涨期权Rho的函数，并且假定无风险收益率增加0.0001（1个基点）
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    Value1 = American_call(S, K, sigma, r, T, N)  # 原二叉树模型计算的期权价值
    Value2 = American_call(S, K, sigma, r+0.0001, T, N)  # 新二叉树模型计算的期权价值
    rho = (Value2 - Value1) / 0.0001  # 计算美式看涨期权的Rho
    return rho

def rho_AmerPut(S,K,sigma,r,T,N):
    """
    定义一个运用N步二叉树模型计算美式看跌期权Rho的函数，依然假定无风险收益率增加0.0001（1个基点）
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    Value1 = American_put(S, K, sigma, r, T, N)  # 原二叉树模型计算的期权价值
    Value2 = American_put(S, K, sigma, r+0.0001, T, N)  # 新二叉树模型计算的期权价值
    rho = (Value2 - Value1) / 0.0001  # 计算美式看跌期权的Rho
    return rho

# 沿用前例信息，同时将期权调整为美式看涨期权、看跌期权，运用自定义函数rho_AmerCall和rho_AmerPut计算美式期权的Rho，并且二叉树模型的步数依然是100步
rho_AmerOpt1 = rho_AmerCall(S = S_ABC, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC,
                            N = step)  # 计算美式看涨期权的Rho
rho_AmerOpt2 = rho_AmerPut(S = S_ABC, K = K_ABC, sigma = sigma_ABC, r = shibor_6M, T = T_ABC,
                           N = step)  # 计算美式看跌期权的Rho
print("农业银行A股美式看涨期权的Rho", round(rho_AmerOpt1, 4))
print("农业银行A股美式看跌期权的Rho", round(rho_AmerOpt2, 4))
# 可看出：美式看涨期权的Rho与欧式看涨期权的Rho很接近；
# 欧式看跌期权Rho的绝对值显著高于美式看跌期权即比起欧式看跌期权，美式看跌期权对无风险收益率变化的敏感性会更低



# 12.6 期权的隐含波动率
# 12.6.1 计算隐含波动率的牛顿迭代法
# 利用牛顿迭代法并运用Python自定义分别计算欧式看涨、看跌期权隐含波动率的函数
def impvol_call_Newton(C, S, K, r, T):
    """
    定义一个运用BSM模型计算欧式看涨期权的隐含波动率的函数，并且使用的迭代方法是牛顿迭代法
    C：代表观察到的看涨期权市场价格
    S：代表基础资产的价格
    K：代表期权的行权价格
    r：代表连续复利的无风险收益率
    T：代表期权的剩余期限（年）
    """
    from numpy import log, exp, sqrt  # 从NumPy模块导入log、exp和sqrt这3个函数
    from scipy.stats import norm  # 从SciPy的子模块stats中导入norm函数

    def call_BSM(S, K, sigma, r, T):  # 定义一个运用BSM模型计算欧式看涨期权价格的函数
        d1 = (log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))  # 计算参数d1
        d2 = d1 - sigma * sqrt(T)  # 计算参数d2
        call = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)  # 计算看涨期权价格
        return call
    sigma0 = 0.2  # 设置一个初始波动率
    diff = C - call_BSM(S, K, sigma0, r, T)  # 计算期权市场价格与BSM模型得到的期权价格的差异值
    i = 0.0001  # 设置一个标量
    while abs(diff) > 0.0001:  # 运用while循环语句
        diff = C - call_BSM(S, K, sigma0, r, T)
        if diff > 0:  # 当差异值大于0时
            sigma0 += i  # 波动率加上一个标量
        else:  # 当差异值小于0时
            sigma0 -= i  # 波动率减去一个标量
    return sigma0

def impvol_put_Newton(P, S, K, r, T):
    """
    定义一个运用BSM模型计算欧式看跌期权的隐含波动率的函数，使用的迭代方法是牛顿迭代法
    C：代表观察到的看跌期权市场价格
    S：代表基础资产的价格
    K：代表期权的行权价格
    r：代表连续复利的无风险收益率
    T：代表期权的剩余期限（年）
    """
    from numpy import log,exp,sqrt
    from scipy.stats import norm

    def put_BSM(S, K, sigma, r, T):
        d1 = (log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        put = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put
    sigma0 = 0.2
    diff = P-put_BSM(S, K, sigma0, r, T)
    i = 0.0001
    while abs(diff) > 0.0001:
        diff = P - put_BSM(S, K, sigma0, r, T)
        if diff > 0:
            sigma0 += i
        else:
            sigma0 -= i
    return sigma0

# 2020年9月1日，上海证券交易所交易的“50ETF购3月3300”期权合约、“50ETF沽3月3300”期权合约的要素以及市场价格（结算价）如P482所示，
# 合约以上证50ETF基金（代码510050）为基础资产，当天上证50ETF基金净值为3.406元，
# 无风险收益率设定为6个月期Shibor并且当天报价为2.847%（连续复利）
# 通过前面自定义函数impvol_call_Newton和impvol_put_Newton，
# 运用牛顿迭代法依次计算“50ETF购3月3300”期权合约、“50ETF沽3月3300”期权合约的隐含波动率
import datetime as dt  # 导入datetime模块

T0 = dt.datetime(2020, 9, 1)  # 隐含波动率的计算日
T1 = dt.datetime(2021, 3, 24)  # 期权到期日
tenor = (T1 - T0).days / 365  # 计算期权的剩余期限（年）

price_call = 0.2826  # 50ETF购3月3300期权合约的价格
price_put = 0.1975  # 50ETF沽3月3300期权合约的价格
price_50ETF = 3.406  # 上证50ETF基金净值
shibor_6M = 0.02847  # 6个月期Shibor
K_50ETF = 3.3  # 期权的行权价格

sigma_call = impvol_call_Newton(C = price_call, S = price_50ETF, K = K_50ETF, r = shibor_6M,
                                T = tenor)  # 计算看涨期权的隐含波动率
print("50ETF购3月3300期权合约的隐含波动率（牛顿迭代法）", round(sigma_call, 4))

sigma_put = impvol_put_Newton(P = price_put, S = price_50ETF, K = K_50ETF, r = shibor_6M,
                              T = tenor)  # 计算看跌期权的隐含波动率
print("50ETF沽3月3300期权合约的隐含波动率（牛顿迭代法）", round(sigma_put, 4))
# 可看出：“50ETF购3月3300”期权合约的隐含波动率为19.5%，“50ETF沽3月3300”期权合约的隐含波动率为27.19%，
# 即期权多头以购买看跌期权为主，期权空头则以卖出看涨期权为主

# 12.6.2 计算隐含波动率的二分查找法
# 利用二分查找法并运用Python自定义分别计算欧式看涨、看跌期权隐含波动率的函数
def impvol_call_Binary(C, S, K, r, T):
    """
    定义一个运用BSM模型计算欧式看涨期权隐含波动率的函数，并且使用的迭代方法是二分查找法
    C：代表观察到的看跌期权市场价格
    S：代表基础资产的价格
    K：代表期权的行权价格
    r：代表连续复利的无风险收益率
    T：代表期权的剩余期限（年）
    """
    from numpy import log, exp, sqrt
    from scipy.stats import norm

    def call_BSM(S, K, sigma, r, T):  # 定义一个运用BSM模型计算看涨期权价格的函数
        d1 = (log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        call = S*norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        return call
    sigma_min = 0.001  # 设置初始的最小隐含波动率
    sigma_max = 1.000  # 设置初始的最大隐含波动率
    sigma_mid = (sigma_min + sigma_max) / 2  # 计算初始的平均隐含波动率
    call_min = call_BSM(S, K, sigma_min, r, T)  # 初始最小隐含波动率对应的期权价格（期权价格初始下限）
    call_max = call_BSM(S, K, sigma_max, r, T)  # 初始最大隐含波动率对应的期权价格（期权价格初始上限）
    call_mid = call_BSM(S, K, sigma_mid, r, T)  # 初始平均隐含波动率对应的期权价格（期权价格初始均值）
    diff = C - call_mid  # 期权市场价格与BSM模型得到的期权价格初始均值的差异值
    if C < call_min or C > call_max:  # 期权市场价格小于期权价格初始下限或大于期权价格初始上限
        print("Error")  # 报错
    while abs(diff) > 1e-6:  # 当差异值的绝对值大于0.000001
        diff = C-call_BSM(S, K, sigma_mid, r, T)  # 期权市场价格与平均隐含波动率对应的期权价格的差异值
        sigma_mid = (sigma_min + sigma_max) / 2  # 计算新的平均隐含波动率
        call_mid = call_BSM(S, K, sigma_mid, r, T)  # 新的平均隐含波动率对应的期权新价格
        if C > call_mid:  # 当期权市场价格大于期权新价格时
            sigma_min = sigma_mid  # 最小隐含波动率赋值为新的平均隐含波动率
        else:  # 当期权市场价格小于期权新价格时
            sigma_max = sigma_mid  # 最大隐含波动率赋值为新的平均隐含波动率
    return sigma_mid

def impvol_put_Binary(P, S, K, r, T):
    """
    定义一个运用BSM模型计算欧式看跌期权隐含波动率的函数，且使用二分查找法迭代
    P: 代表观察到的看跌期权市场价格
    S: 代表基础资产的价格
    K: 代表期权的行权价格
    r: 代表连续复利的无风险收益率
    T: 代表期权的剩余期限（年）
    """
    from numpy import log, exp, sqrt
    from scipy.stats import norm

    def put_BSM(S, K, sigma, r, T):  # 定义一个运用BSM模型计算欧式看跌期权价格的函数
        d1 = (log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        put = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put
    sigma_min = 0.001
    sigma_max = 1.000
    sigma_mid = (sigma_min + sigma_max) / 2
    put_min = put_BSM(S, K, sigma_min, r, T)
    put_max = put_BSM(S, K, sigma_max, r, T)
    put_mid = put_BSM(S, K, sigma_mid, r, T)
    diff = P - put_mid
    if P < put_min or P > put_max:
        print("Error")
    while abs(diff) > 1e-6:
        diff = P - put_BSM(S, K, sigma_mid, r, T)
        sigma_mid = (sigma_min + sigma_max) / 2
        put_mid = put_BSM(S, K, sigma_mid, r, T)
        if P > put_mid:
            sigma_min = sigma_mid
        else:
            sigma_max = sigma_mid
    return sigma_mid

# 沿用前例信息，通过自定义函数impvol_call_Binary和impvol_put_Binary运用二分查找法依次计算“50ETF购3月3300”期权合约、“50ETF沽3月3300”期权合约的隐含波动率
sigma_call = impvol_call_Binary(C = price_call, S = price_50ETF, K = K_50ETF, r = shibor_6M,
                                T = tenor)  # 计算看涨期权的隐含波动率
print("50ETF购3月3300期权合约的隐含波动率（二分查找法）", round(sigma_call, 4))

sigma_put = impvol_put_Binary(P = price_put, S = price_50ETF, K = K_50ETF, r = shibor_6M,
                              T = tenor)  # 计算看跌期权的隐含波动率
print("50ETF沽3月3300期权合约的隐含波动率（二分查找法）", round(sigma_put, 4))

# 12.6.3 波动率微笑
# 以2021年6月23日到期的、不同行权价格的上证50ETF认沽期权合约在2020年12月31日的结算价数据作为分析对象，并且选择11只认沽（看跌）期权，具体信息详见P485.
# 当天的上证50ETF基金净值为3.635元，无风险收益率依然运用6个月期Shibor并且报价为2.838%
# 第1步：在Python中输入相关的变量，并计算上证50ETF认沽期权（看跌期权）的隐含波动率，需要运用for语句
S_Dec31 = 3.635  # 2020年12月31日上证50ETF基金净值
R_Dec31 = 0.02838  # 2020年12月31日6个月期Shibor

T2 = dt.datetime(2020, 12, 31)  # 隐含波动率的计算日
T3 = dt.datetime(2021, 6, 23)  # 期权到期日
tenor1 = (T3 - T2).days / 365  # 计算期权的剩余期限（年）

Put_list = np.array([0.0202, 0.0306, 0.0458, 0.0671, 0.0951, 0.1300, 0.1738, 0.2253, 0.2845, 0.3540, 0.4236])  # 上证50ETF认沽期权结算价

K_list1 = np.array([3.0000, 3.1000, 3.2000, 3.3000, 3.4000, 3.5000, 3.6000, 3.7000, 3.8000, 3.9000, 4.0000])  # 期权的行权价格

n1 = len(K_list1)  # 不同行权价格的看跌期权合约数量

sigma_list1 = np.zeros_like(Put_list)  # 构建存放看跌期权隐含波动率的初始数组

for i in np.arange(n1):
    sigma_list1[i] = impvol_put_Newton(P = Put_list[i], S = S_Dec31, K = K_list1[i], r = R_Dec31,
                                       T = tenor1)  # 运用牛顿迭代法计算看跌期权的隐含波动率

# 第2步：将行权价格与隐含波动率的关系可视化，也就是绘制出波动率微笑曲线
plt.figure(figsize = (9, 6))
plt.plot(K_list1, sigma_list1, "b-", lw = 2.5)
plt.xlabel(u"期权的行权价格", fontsize = 13)
plt.ylabel(u"隐含波动率", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"期权的行权价格与上证50ETF认沽期权隐含波动率", fontsize = 13)
plt.grid()
plt.show()
# 可看出：当行权价格越接近基础资产价格（3.635元）时，期权的隐含波动率基本上越低；当行权价格越远离基础资产价格时，期权的隐含波动率越高

# 12.6.4 波动率斜偏
# 针对在深圳证券交易所挂牌并且在2021年3月24日到期的沪深300ETF认购期权合约，以2020年9月30日的结算价数据作为分析依据，并且选择10只不同行权价格的认购（看涨）期权，具体信息见P487。
# 当天的沪深300ETF基金（代码159919）净值为4.5848元，无风险收益率运用6个月Shibor并且报价为2.691%
# 第1步：在Python中输入相关的变量，计算沪深300ETF认购期权的隐含波动率，需要运用for语句
S_Sep30 = 4.5848  # 2020年9月30日沪深300ETF基金净值
R_Sep30 = 0.02691  # 2020年9月30日6个月期Shibor

T4 = dt.datetime(2020, 9, 30)  # 隐含波动率的计算日
T5 = dt.datetime(2021, 3, 24)  # 期权到期日
tenor2 = (T5 - T4).days / 365  # 计算期权的剩余期限（年）

Call_list = np.array([0.4660, 0.4068, 0.3529, 0.3056, 0.2657, 0.2267, 0.1977, 0.1707, 0.1477,
                      0.1019])  # 沪深300ETF认购期权结算价

K_list2 =  np.array([4.2000, 4.3000, 4.4000, 4.5000, 4.6000, 4.7000, 4.8000, 4.9000, 5.0000,
                     5.25000])  # 期权的行权价格

n2 = len(K_list2)  # 不同行权价格的看涨期权合约数量

sigma_list2 = np.zeros_like(Call_list)  # 构建存放看涨期权隐含波动率的初始数组

for i in np.arange(n2):
    sigma_list2[i] = impvol_call_Binary(C = Call_list[i], S = S_Sep30, K = K_list2[i], r = R_Sep30,
                                        T = tenor2)  # 运用二分查找法计算看涨期权的隐含波动率

# 第2步：将行权价格与隐含波动率的关系可视化，即绘制出波动率向上斜偏曲线
plt.figure(figsize = (9, 6))
plt.plot(K_list2, sigma_list2, "r-", lw = 2.5)
plt.xlabel(u"期权的行权价格", fontsize = 13)
plt.ylabel(u"隐含波动率", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"期权的行权价格与沪深300ETF认购期权隐含波动率", fontsize = 13)
plt.grid()
plt.show()
# 可看出：对于行权价格较低的期权即深度实值看涨期权，其隐含波动率较低；对于行权价格较高的期权即深度虚值看涨期权，其隐含波动率则较高
# 可能会出现期权波动率不存在的情况，即当基础资产的波动率取极小值（比如0.0001%甚至更小）时，
# 通过BSM模型计算得到的欧式期权价格仍高于期权市场价格，此时无法计算出欧式期权的隐含波动率

