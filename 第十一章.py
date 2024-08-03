# 第十一章
# 11.1 A股期权市场简介
# 11.1.2 股票期权合约
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["FangSong"]
plt.rcParams["axes.unicode_minus"] = False
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

option_300ETF = pd.read_excel("D:/Python/300ETF购12月3500合约每日价格数据.xlsx", sheet_name = "Sheet1", header = 0,
                              index_col = 0) # 导入外部数据

option_300ETF.plot(figsize = (9, 6), title = u"300ETF购12月3500合约的日交易价格走势图", grid = True, fontsize = 13) # 可视化
plt.ylabel(u'金额', fontsize = 11) # 增加纵坐标标签

# 11.1.3 股指期权合约
option_HS300 = pd.read_excel("D:/Python/沪深300股指沽12月4000合约每日价格数据.xlsx", sheet_name = "Sheet1", header = 0,
                             index_col = 0) # 导入外部数据
option_HS300.plot(figsize = (9, 6), title = u"沪深300股指沽12月4000合约的日交易价格走势图", grid = True, fontsize = 12) # 可视化
plt.ylabel(u'金额', fontsize = 11)



# 11.2 期权类型与到期盈亏
# 11.2.2 看涨期权的到期盈亏
# 例11.1（P401）假定A投资者在2020年8月20日买入基础资产为10000股工商银行A股、行权价格为5.3元/股的欧式看涨期权，
# 购买时工商银行A股（代码为601398）股价恰好是5元/股，期权到期日为6个月以后（2021年2月20日），期权费是0.1元/股，
# A投资者最初投资10000*0.1=1000元，即一份看涨期权的期权费为1000元
"""
情形1：在期权到期日，股价低于行权价格5.3元/股，期权不会被行使
情形2：在期权到期日，股价高于行权价格5.3元/股，期权会被行使
情形3：在期权到期日，股价等于行权价格5.3元/股，期权被行使与不被行使是无差异的
"""
S = np.linspace(4, 7, 200) # 期权到期时工商银行A股股价的等差数列
K_call = 5.3 # 看涨期权的行权价格
C = 0.1 # 看涨期权的期权费
N = 10000 # 一份看涨期权对应基础资产工商银行A股的数量

profit1_call = N * np.maximum(S - K_call, 0) # 期权到期时不考虑期权费的收益
profit2_call = N * np.maximum(S - K_call - C, -C) # 期权到期时考虑期权费以后的收益

plt.figure(figsize = (9, 6))
plt.subplot(1, 2, 1) # 第1个子图
plt.plot(S, profit1_call, "b-", label = u"不考虑期权费的期权多头收益", lw = 2.5)
plt.plot(S, profit2_call, "b--", label = u"考虑期权费的期权多头收益", lw = 2.5)
plt.xlabel(u"工商银行A股价格", fontsize = 13)
plt.xticks(fontsize = 13)
plt.ylabel(u"期权盈亏", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"看涨期权到期日多头的盈亏", fontsize = 13)
plt.legend(fontsize = 12)
plt.grid()
plt.subplot(1, 2, 2) # 第2个子图
plt.plot(S, -profit1_call, "r-", label = u"不考虑期权费的期权空头收益", lw = 2.5)
plt.plot(S, -profit2_call, "r--", label = u"考虑期权费的期权空头收益", lw = 2.5)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"看涨期权到期日空头的盈亏", fontsize = 13)
plt.legend(fontsize = 12)
plt.grid()
plt.show()

# 11.2.3 看跌期权的到期盈亏
# 假定B投资者于2020年8月11日买入基础资产为10000股工商银行A股、行权价格为5.1元/股的欧式看跌期权，
# 购买时工商银行A股股价也是5元/股，期权到期日为12个月以后（2021年8月11日），期权费是0.2元/股，
# B投资者最初投资10000*0.2=2000元，也就是一份看跌期权的期权费是2000元
"""
情形1：在期权到期日，股价低于行权价格5.1元/股，期权会被行使
情形2：在期权到期日，股价高于行权价格5.1元/股，期权不会被行使
情形3：在期权到期日，股价等于行权价格5.1元/股，期权被行使与不被行使是无差异的
"""
K_put = 5.1 # 看跌期权的行权价格
P = 0.2 # 看跌期权的期权费

profit1_put = N * np.maximum(K_put - S, 0) # 期权到期时不考虑期权费的收益
profit2_put = N * np.maximum(K_put - S - P, -P) # 期权到期时考虑期权费以后的收益

plt.figure(figsize = (9, 6))
plt.subplot(1, 2, 1) # 第1个子图
plt.plot(S, profit1_put, "b-", label = u"不考虑期权费的期权多头收益", lw = 2.5)
plt.plot(S, profit2_put, "b--", label = u"考虑期权费的期权多头收益", lw = 2.5)
plt.xlabel(u"工商银行A股价格", fontsize = 13)
plt.xticks(fontsize = 13)
plt.ylabel(u"期权盈亏", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"看跌期权到期日多头的盈亏", fontsize = 13)
plt.legend(fontsize = 12)
plt.grid()
plt.subplot(1, 2, 2) # 第2个子图
plt.plot(S, -profit1_put, "r-", label = u"不考虑期权费的期权空头收益", lw = 2.5)
plt.plot(S, -profit2_put, "r--", label = u"考虑期权费的期权空头收益", lw = 2.5)
plt.xlabel(u"工商银行A股价格", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"看跌期权到期日空头的盈亏", fontsize = 13)
plt.legend(fontsize = 12)
plt.grid()
plt.show()

# 11.2.4 看跌-看涨平价关系式
# 用Python自定义一个通过看跌-看涨平价关系式计算欧式看涨、看跌期权价格的函数
def option_parity(opt, c, p, S, K, r, T):
    """
    定义一个通过看跌-看涨平价关系式计算欧式看涨、看跌期权价格的函数
    opt：代表需要计算的欧式期权类型，输入opt='call'表示计算看涨期权价格，输入其他则表示计算看跌期权价格
    c：代表看涨期权价格，如果计算看涨期权价格，则输入c='Na'
    p：代表看跌期权价格，如果计算看跌期权价格，则输入p='Na'
    S：代表期权基础资产的价格
    K：代表期权的行权价格
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    """
    from numpy import exp # 导入NumPy模块的exp函数
    if opt == "call": # 针对欧式看涨期权
        value = p + S - K * exp(-r * T) # 计算欧式看涨期权价格
    else: # 针对欧式看跌期权
        value = c + K * exp(-r * T) - S # 计算欧式看跌期权价格
    return value

# 2020年8月20日，以工商银行A股为基础资产、行权价格为5.2元/股、期限为3个月的欧式看涨期权的市场报价是0.15元，欧式看跌期权的市场报价是0.3元。
# 当天工商银行A股收盘价为5元/股，以3个月期Shibor作为无风险收益率，当天报价是2.601%并且是连续复利。
# 通过看跌-看涨平价关系式判断期权报价是否合理，如果报价不满足看跌-看涨平价关系式则如何实施套利
price_call = 0.15 # 看涨期权市场报价
price_put = 0.3 # 看跌期权市场报价
S_ICBC = 5.0 # 工商银行A股价格
K_ICBC = 5.2 # 期权行权价格
shibor = 0.02601 # 3个月期Shibor
tenor = 3 / 12 # 期权期限（年）

value_call = option_parity(opt = "call", c = "Na", p = price_put, S = S_ICBC, K = K_ICBC, r = shibor,
                           T = tenor) # 计算看涨期权价格
value_put = option_parity(opt = "put", c = price_call, p = "Na", S = S_ICBC, K = K_ICBC, r = shibor,
                          T = tenor) # 计算看跌期权价格
print("运用看跌-看涨平价关系式得出欧式看涨期权价格", round(value_call, 4))
print("运用看跌-看涨平价关系式得出欧式看跌期权价格", round(value_put, 4))
# 看涨期权被高估，看跌期权被低估，可通过持有看涨期权的空头头寸并卖空零息债券（即卖空A投资组合）同时持有看跌期权的多头头寸并买入基础资产（即买入B投资组合），从而实现无风险套利

# 11.3 欧式期权定价——布莱克-斯科尔斯-默顿模型
# 11.3.1 模型介绍
# 通过Python自定义一个运用布莱克-斯科尔斯-默顿模型计算欧式看涨、看跌期权价格的函数
def option_BSM(S, K, sigma, r, T, opt):
    """
    定义一个运用布莱克-斯科尔斯-默顿模型计算欧式期权价格的函数
    S：代表期权基础资产的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    opt：代表期权类型，输入opt="call"表示看涨期权，输入其他则表示看跌期权
    """
    from numpy import log, exp, sqrt # 从NumPy模块导入log、exp、sqrt这3个函数
    from scipy.stats import norm # 从SciPy的子模块stats导入norm函数
    d1 = (log(S / K) + (r + pow(sigma, 2) / 2) * T) / (sigma * sqrt(T)) # 计算参数d1
    d2 = d1 - sigma * sqrt(T) # 计算参数d2
    if opt == "call": # 针对欧式看涨期权
        value = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2) # 计算期权价格
    else: # 针对欧式看跌期权
        value = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1) # 计算期权价格
    return value

# 沿用前例信息，即考虑基础资产是工商银行A股股票、期限为3个月、期权的行权价格为5.2元/股的欧式看涨、看跌期权，
# 2020年8月20日股票收盘价是5.0元/股，无风险收益率运用3个月期Shibor并且等于2.601%，股票收益率的年化波动率是20.5%，
# 运用布莱克-斯科尔斯-默顿模型计算当天期权的价格
sigma_ICBC = 0.205 # 工商银行A股收益率的年化波动率

call_BSM = option_BSM(S = S_ICBC, K = K_ICBC, sigma = sigma_ICBC, r = shibor, T = tenor, opt = "call") # 计算看涨期权价格
put_BSM = option_BSM(S = S_ICBC, K = K_ICBC, sigma = sigma_ICBC, r = shibor, T = tenor, opt = "put") # 计算看跌期权价格
print("运用布莱克-斯科尔斯-默顿模型得到欧式看涨期权价格", round(call_BSM, 4))
print("运用布莱克-斯科尔斯-默顿模型得到欧式看跌期权价格", round(put_BSM, 4))
# 通过布莱克-斯科尔斯-默顿模型不难发现，存在5个影响期权价格的变量：
# 1.当前基础资产价格S；2.期权的行权价格K；3.期权期限T；4.基础资产的波动率sigma；5.无风险收益率r

# 11.3.2 期权价格与基础资产价格的关系
# 沿用前例的工商银行股票期权信息，对股票价格设定在一个取值是在[4.0, 6.0]区间的等差数列，其他变量的取值保持不变，
# 运用布莱克-斯科尔斯-默顿模型对期权进行定价，从而模拟基础资产价格与期权价格的关系
S_list = np.linspace(4.0, 6.0, 100) # 工商银行A股股价的等差数列

call_list1 = option_BSM(S = S_list, K = K_ICBC, sigma = sigma_ICBC, r = shibor, T = tenor, opt = "call")
put_list1 = option_BSM(S = S_list, K = K_ICBC, sigma = sigma_ICBC, r = shibor, T = tenor, opt = "put")

plt.figure(figsize = (9, 6))
plt.plot(S_list, call_list1, "b-", label = u"欧式看涨期权", lw = 2.5)
plt.plot(S_list, put_list1, "r-", label = u"欧式看跌期权", lw = 2.5)
plt.xlabel(u"工商银行A股股价", fontsize = 13)
plt.xticks(fontsize = 13)
plt.ylabel(u"期权价格", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"工商银行A股股价（基础资产价格）与期权价格的关系图", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 可看出：随着基础资产价格上升，看涨期权价格会上升；看跌期权价格下跌；基础资产价格与期权价格之间为非线性关系

# 11.3.3 期权价格与行权价格的关系
# 沿用前例的工商银行股票期权信息，对期权行权价格设定在一个取值是在[4.2, 6.2]区间的等差数列，其他变量的取值保持不变
K_list = np.linspace(4.2, 6.2, 100) # 期权行权价格的等差数列

call_list2 = option_BSM(S = S_ICBC, K = K_list, sigma = sigma_ICBC, r = shibor, T = tenor, opt = "call") # 计算看涨期权价格
put_list2 = option_BSM(S = S_ICBC, K = K_list, sigma = sigma_ICBC, r = shibor, T = tenor, opt = "put") # 计算看跌期权价格

plt.figure(figsize = (9, 6))
plt.plot(K_list, call_list2, "b-", label = u"欧式看涨期权", lw = 2.5)
plt.plot(K_list, put_list2, "r-", label = u"欧式看跌期权", lw = 2.5)
plt.xlabel(u"行权价格", fontsize = 13)
plt.xticks(fontsize = 13)
plt.ylabel(u"期权价格", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"行权价格与期权价格的关系图", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 可看出：随着基础资产价格上升，看涨期权价格会上升；看跌期权价格下跌；基础资产价格与期权价格之间为非线性关系；此曲线是前例曲线的镜像反映

# 11.3.4 期权价格与波动率的关系
# 沿用前例的工商银行股票期权信息，对股票收益率的波动率设定在一个取值是在[1%, 30%]区间的等差数列，其他变量的取值保持不变
sigma_list = np.linspace(0.01, 0.3, 100) # 波动率的等差数列

call_list3 = option_BSM(S = S_ICBC, K = K_ICBC, sigma = sigma_list, r = shibor, T = tenor, opt = "call") # 计算看涨期权价格
put_list3 = option_BSM(S = S_ICBC, K = K_ICBC, sigma = sigma_list, r = shibor, T = tenor, opt = "put") # 计算看跌期权价格

plt.figure(figsize = (9, 6))
plt.plot(sigma_list, call_list3, "b-", label = u"欧式看涨期权", lw = 2.5)
plt.plot(sigma_list, put_list3, "r-", label = u"欧式看跌期权", lw = 2.5)
plt.xlabel(u"波动率", fontsize = 13)
plt.xticks(fontsize = 13)
plt.ylabel(u"期权价格", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"波动率与期权价格的关系图", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 可看出：随着基础资产收益率的波动率上升，看涨期权和看跌期权的价格会上升；基础资产收益率的波动率与期权价格之间为非线性关系；当波动率很小时，期权价格对波动率就变得不敏感

# 11.3.5 期权价格与无风险收益率的关系
# 沿用前例的工商银行股票期权信息，对无风险收益率设定在一个取值是在[0.01, 0.1]区间的等差数列，其他变量的取值保持不变
shibor_list = np.linspace(0.01, 0.10, 100) # 无风险收益率的等差数列

call_list4 = option_BSM(S = S_ICBC, K = K_ICBC, sigma = sigma_ICBC, r = shibor_list, T = tenor, opt = "call") # 计算看涨期权价格
put_list4 = option_BSM(S = S_ICBC, K = K_ICBC, sigma = sigma_ICBC, r = shibor_list, T = tenor, opt = "put") # 计算看跌期权价格

plt.figure(figsize = (9, 6))
plt.plot(shibor_list, call_list4, "b-", label = u"欧式看涨期权", lw = 2.5)
plt.plot(shibor_list, put_list4, "r-", label = u"欧式看跌期权", lw = 2.5)
plt.xlabel(u"无风险收益率", fontsize = 13)
plt.xticks(fontsize = 13)
plt.ylabel(u"期权价格", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"无风险收益率与期权价格的关系图", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 可看出：随着无风险收益率上升，看涨期权的价格上升；看跌期权价格下降；
# 1.无风险收益率增加则用于贴现的利率上升，期权行权价格的现值下降，增加看涨期权价格，减少看跌期权价格；
# 2.投资基础资产要占用投资者一定资金，对应相同规模基础资产的期权投入资金少，利率高则购买基础资产占用的资金成本高，期权吸引力越大；两者叠加，看涨期权价格上涨，看跌期权价格下跌

# 11.3.6 期权价格与期权期限的关系
# 沿用前例的工商银行股票期权信息，对期权期限设定在一个取值是在[0.1, 3.0]区间的等差数列，其他变量的取值保持不变
tenor_list = np.linspace(0.1, 3.0, 100) # 期权期限的等差数列

call_list5 = option_BSM(S = S_ICBC, K = K_ICBC, sigma = sigma_ICBC, r = shibor, T = tenor_list, opt = "call") # 计算看涨期权价格
put_list5 = option_BSM(S = S_ICBC, K = K_ICBC, sigma = sigma_ICBC, r = shibor, T = tenor_list, opt = "put") # 计算看跌期权价格

plt.figure(figsize = (9, 6))
plt.plot(tenor_list, call_list5, "b-", label = u"欧式看涨期权", lw = 2.5)
plt.plot(tenor_list, put_list5, "r-", label = u"欧式看跌期权", lw = 2.5)
plt.xlabel(u"期权期限", fontsize = 13)
plt.xticks(fontsize = 13)
plt.ylabel(u"期权价格", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"期权期限与期权价格的关系图", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 可看出：期权期限上升，看涨期权和看跌期权的价格上升；期权期限上升，看涨期权价格增幅高于看跌期权价格增幅；
# 期权期限低于1.5年时，看涨期权价格低于看跌期权价格，期权期限高于1.5年时，看涨期权价格高于看跌期权

# 11.3.7 内在价值与时间价值
# 沿用前例工商银行股票期权信息，假定2020年8月20日（期权初始日），工商银行A股收盘价的取值是处于[4.7, 6.0]区间的等差数列
# 第1步：运用BSM模型计算2020年8月20日对应不同基础资产价格的期权价格，同时计算当天期权假定被立刻行权所产生的收益并且不考虑期权费
S_list = np.linspace(4.7, 6, 200) # 设定工商银行A股股价的等差数列

price_call = option_BSM(S = S_list, K = K_ICBC, sigma = sigma_ICBC, r = shibor, T = tenor, opt = "call") # 计算看涨期权价格
price_put = option_BSM(S = S_list, K = K_ICBC, sigma = sigma_ICBC, r = shibor, T = tenor, opt = "put") # 计算看跌期权价格

profit_call = np.maximum(S_list - K_ICBC, 0) # 计算看涨期权被行权所产生的收益（不考虑期权费）
profit_put = np.maximum(K_ICBC - S_list, 0) # 计算看跌期权被行权所产生的收益（不考虑期权费）

# 第2步：针对第1步计算得到的期权价格与盈亏数据进行可视化，进而便于比较分析
plt.figure(figsize = (9, 7))
plt.subplot(2, 1, 1) # 第1个子图
plt.plot(S_list, price_call, "b-", label = u"欧式看涨期权价格", lw = 2.5)
plt.plot(S_list, profit_call, "r-", label = u"欧式看涨期权被行权的收益", lw = 2.5)
plt.xticks(fontsize = 13)
plt.ylabel(u"看涨期权价格或盈亏", fontsize = 13)
plt.yticks(fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.subplot(2, 1, 2) # 第2个子图
plt.plot(S_list, price_put, "b-", label = u"欧式看跌期权价格", lw = 2.5)
plt.plot(S_list, profit_put, "r-", label = u"欧式看跌期权被行权的收益", lw = 2.5)
plt.xlabel(u"股票价格", fontsize = 13)
plt.xticks(fontsize = 13)
plt.ylabel(u"看跌期权价格或盈亏", fontsize = 13)
plt.yticks(fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 可看出：无论看涨期权还是看跌期权，2020年8月20日期权价格高于当天期权被立刻行权所产生的收益



# 11.4 欧式期权定价——二叉树模型
# 11.4.1 一步二叉树模型
# 通过Python自定义一个运用一步二叉树模型计算欧式期权价值的函数
def BTM_1step(S, K, u, d, r, T, types):
    """
    定义一个运用一步二叉树模型计算欧式期权价值的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    u：代表基础资产价格上涨时价格变化比例
    d：代表基础资产价格下跌时价格变化比例
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    types：代表期权类型，输入types=="call"表示欧式看涨期权，输入其他则表示欧式看跌期权
    """
    from numpy import exp, maximum # 从NumPy模块中导入exp、maximum函数
    p = (exp(r * T) - d) / (u - d) # 基础资产价格上涨的概率
    Cu = maximum(S * u - K, 0) # 期权到期时基础资产价格上涨对应的期权价值
    Cd = maximum(S * d - K, 0) # 期权到期时基础资产价格下跌时对应的期权价值
    call = (p * Cu + (1 - p) * Cd) * exp(-r * T) # 初始日的看涨期权价值
    put = call + K * exp(-r * T) - S # 初始日的看跌期权价值（运用看跌-看涨平价关系式）
    if types == "call": # 针对看涨期权
        value = call # 期权价值等于看涨期权价值
    else: # 针对看跌期权
        value = put # 期权价值等于看跌期权价值
    return value

# 通过自定义函数BTM_1step计算前例的期权价值
S_ICBC = 6 # 工商银行A股股价
K_ICBC = 5.7 # 期权的行权价格
up = 1.1 # 在1年后股价上涨情形中的股价变化比例
down = 0.9 # 在1年后股价下跌情形中的股价变化比例
R = 0.024 # 无风险收益率
tenor = 1.0 # 期权期限（年）

value_call = BTM_1step(S = S_ICBC, K = K_ICBC, u = up, d = down, r = R, T = tenor, types = "call")
print("2020年1月3日工商银行股票看涨期权价值", round(value_call, 3))

value_put = BTM_1step(S = S_ICBC, K = K_ICBC, u = up, d = down, r = R, T = tenor, types = "put")
print("2020年1月3日工商银行股票看跌期权价值", round(value_put, 3))

# 11.4.2 两步二叉树模型
# 通过Python自定义一个运用两步二叉树模型计算欧式期权价值的函数
def BTM_2step(S, K, u, d, r, T, types):
    """
    定义一个运用两步二叉树模型计算欧式期权价值的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    u：代表基础资产价格上涨时价格变化比例
    d：代表基础资产价格下跌时价格变化比例
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    types：代表期权类型，输入types=="call"表示欧式看涨期权，输入其他则表示欧式看跌期权
    """
    from numpy import exp, maximum # 从NumPy模块中导入exp、maximum函数
    t = T / 2 # 每一步步长期限（年）
    p = (exp(r * t) - d) / (u - d) # 基础资产价格上涨的概率
    Cuu = maximum(pow(u, 2) * S - K, 0) # 期权到期时基础资产价格两次上涨对应的期权价值
    Cud = maximum(S * u * d - K, 0) # 期权到期时基础资产价格一涨一跌对应的期权价值
    Cdd = maximum(pow(d, 2) * S - K, 0) # 期权到期时基础资产价格两次下跌对应的期权价值
    call = (pow(p, 2) * Cuu + 2 * p * (1 - p) * Cud + pow(1 - p, 2) * Cdd) * np.exp(-r * T) # 看涨期权价值
    put = call + K * exp(-r * T) - S # 看跌期权价值（运用看跌-看涨平价关系式）
    if types == "call": # 针对看涨期权
        value = call # 期权价值等于看涨期权价值
    else: # 针对看跌期权
        value = put # 期权价值等于看跌期权价值
    return value

# 计算前例看涨期权价值
tenor_new = 2 # 期权期限

value_call_2Y = BTM_2step(S = S_ICBC, K = K_ICBC, u = up, d = down, r = R, T = tenor_new, types = "call")
print("2020年1月3日工商银行股票看涨期权价值", round(value_call_2Y, 4))

# 将前例的看涨期权调整为看跌期权，其他信息均保持不变
value_put_2Y = BTM_2step(S = S_ICBC, K = K_ICBC, u = up, d = down, r = R, T = tenor_new, types = "put")
print("2020年1月3日工商银行股票看跌期权价值", round(value_put_2Y, 4))

# 11.4.3 N步二叉树模型
# 通过Python自定义一个运用N步二叉树模型计算欧式期权价值的函数
def BTM_Nstep(S, K, sigma, r, T, N, types):
    """
    定义一个运用N步二叉树模型计算欧式期权价值的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    types：代表期权类型，输入types=="call"，表示欧式看涨期权，输入其他则表示欧式看跌期权
    """
    from math import factorial # 导入math模块的阶乘函数
    from numpy import exp, maximum, sqrt # 导入NumPy模块的exp、maximum和sqrt函数
    t = T / N # 计算每一步步长期限（年）
    u = exp(sigma * sqrt(t)) # 计算基础资产价格上涨时的比例
    d = 1 / u # 计算基础资产价格下跌时的比例
    p = (exp(r * t) - d) / (u - d) # 计算基础资产价格上涨的概率
    N_list = range(0, N+1) # 创建从0到N的自然数数列
    A = [] # 设置一个空的列表
    for j in N_list:
        C_Nj = maximum(S * pow(u, j) * pow(d, N-j) - K, 0) # 计算期权到期日某节点的期权价值
        Num = factorial(N) / (factorial(j) * factorial(N-j)) # 到达到期日该节点的实现路径数量
        A.append(Num * pow(p, j) * pow(1-p, N-j) * C_Nj) # 在列表尾部每次增加一个新元素
    call = exp(-r * T) * sum(A) # 计算看涨期权的期初价值，运用式（11-32）
    put = call + K * np.exp(-r * T) - S # 计算看跌期权的期初价值
    if types == "call": # 针对看涨期权
        value = call # 期权价值等于看涨期权的价值
    else: # 针对看跌期权
        value = put # 期权价值等于看跌期权的价值
    return value

# 沿用前例信息，需要运用N步二叉树模型计算2020年1月3日基础资产是工商银行A股股票、期限为1年、行权价格为5.7元/股的欧式看涨期权价值。
# 工商银行A股在2020年1月3日的股价是6元/股，无风险收益率（连续复利）是2.4%
# 第1步：导入存放2017年至2019年工商银行A股日收盘价数据的Excel文件，并计算波动率
P_ICBC = pd.read_excel("D:/Python/工商银行A股日收盘价（2017-2019年）.xls", sheet_name = "Sheet1", header = 0,
                       index_col = 0) # 从外部导入工商银行A股日收盘价的数据

R_ICBC = np.log(P_ICBC / P_ICBC.shift(1)) # 计算工商银行股票每日收益率

Sigma_ICBC = np.sqrt(252) * np.std(R_ICBC) # 计算工商银行股票年化波动率
Sigma_ICBC = float(Sigma_ICBC) # 转换为浮点型数据
print("工商银行A股年化波动率", round(Sigma_ICBC, 4))

# 第2步：分别运用步长等于每月（12步二叉树）、每周（52步二叉树）以及每个交易日（252步二叉树）的二叉树模型计算2020年1月3日的期权价值
N_month = 12 # 步长等于每月
N_week = 52 # 步长等于每周
N_day = 252 # 步长等于每个交易日

Call_value1 = BTM_Nstep(S = S_ICBC, K = K_ICBC, sigma = Sigma_ICBC, r = R, T = tenor, N = N_month,
                        types = "call") # 运用12步二叉树模型计算期权价值
Call_value2 = BTM_Nstep(S = S_ICBC, K = K_ICBC, sigma = Sigma_ICBC, r = R, T = tenor, N = N_week,
                        types = "call") # 运用52步二叉树模型计算期权价值
Call_value3 = BTM_Nstep(S = S_ICBC, K = K_ICBC, sigma = Sigma_ICBC, r = R, T = tenor, N = N_day,
                        types = "call") # 运用252步二叉树模型计算期权价值
print("运用12步二叉树模型（步长等于每月）计算2020年1月3日期权价值", round(Call_value1, 4))
print("运用52步二叉树模型（步长等于每周）计算2020年1月3日期权价值", round(Call_value2, 4))
print("运用252步二叉树模型（步长等于每个交易日）计算2020年1月3日期权价值", round(Call_value3, 4))
# 可看出：当步数比较少时，二叉树模型给出的期权价值结果时比较粗糙的，但是随着步数的增加，二叉树模型给出的期权价值结果会越来越精确

# BSM模型与二叉树模型的关系
# 假定在2020年8月18日，市场推出以建设银行A股股票（代码为601939）作为基础资产、期限为1年、行权价格为6.6元/股的欧式看涨期权，
# 当天股票收盘价为6.32元/股，用1年期国债到期收益率作为无风险收益率（连续复利）并且当天该利率等于2.28%
# 第1步：导入存放2018年至2020年8月18日期间建设银行A股日收盘价数据的Excel文件，运用自定义函数option_BSM计算运用BSM模型得到的期权价值
Price_CCB = pd.read_excel("D:/Python/建设银行A股收盘价（2018年至2020年8月18日）.xlsx", sheet_name = "Sheet1", header = 0,
                          index_col = 0) # 导入建设银行A股日收盘价的数据

R_CCB = np.log(Price_CCB / Price_CCB.shift(1)) # 计算建设银行股票每日收益率

Sigma_CCB = np.sqrt(252) * np.std(R_CCB) # 计算建设银行股票年化波动率
Sigma_CCB = float(Sigma_CCB) # 转换为浮点型数据
print("建设银行A股年化波动率", round(Sigma_CCB, 4))

S_CCB = 6.32 # 2020年8月18日建设银行股票收盘价
T_CCB = 1 # 期权期限
R_Aug18 = 0.0228 # 无风险收益率
K_CCB = 6.6 # 期权行权价格

value_BSM = option_BSM(S = S_CCB, K = K_CCB, sigma = Sigma_CCB, r = R_Aug18, T = T_CCB, opt = "call") # 运用BSM模型对期权定价
print("运用BSM模型计算得到建设银行股票看涨期权价值", round(value_BSM, 4))

# 第2步：分别运用10步、50步和250步二叉树模型计算期权的价值
N1 = 10 # 步数等于10
N2 = 50 # 步数等于50
N3 = 250 # 步数等于250

value_BTM_N1 = BTM_Nstep(S = S_CCB, K = K_CCB, sigma = Sigma_CCB, r = R_Aug18, T = T_CCB, N = N1,
                         types = "call") # 运用10步二叉树模型对期权定价
value_BTM_N2 = BTM_Nstep(S = S_CCB, K = K_CCB, sigma = Sigma_CCB, r = R_Aug18, T = T_CCB, N = N2,
                         types = "call") # 运用50步二叉树模型对期权定价
value_BTM_N3 = BTM_Nstep(S = S_CCB, K = K_CCB, sigma = Sigma_CCB, r = R_Aug18, T = T_CCB, N = N3,
                         types = "call") # 运用250步二叉树模型对期权定价
print("运用10步二叉树模型计算得到建设银行股票看涨期权价值", round(value_BTM_N1, 4))
print("运用50步二叉树模型计算得到建设银行股票看涨期权价值", round(value_BTM_N2, 4))
print("运用250步二叉树模型计算得到建设银行股票看涨期权价值", round(value_BTM_N3, 4))
# 可看出：随着步数的增加，计算得到的结果越来越收敛于BSM模型的结果

# 第3步：运用可视化的方法考察二叉树模型的结果如何收敛于BSM模型的结果
N_list = range(1, 151) # 创建1到150的整数列表作为步数

value_BTM_list = np.zeros(len(N_list)) # 创建存放期权价值的初始数组

for i in N_list: # 通过for语句计算不同步数的二叉树模型所得到的期权价值
    value_BTM_list[i-1] = BTM_Nstep(S = S_CCB, K = K_CCB, sigma = Sigma_CCB, r = R_Aug18, T = T_CCB, N = i,
                                    types = "call")

value_BSM_list = value_BSM * np.ones(len(N_list)) # 创建运用BSM模型计算得到的期权价值的数组

plt.figure(figsize = (9, 6))
plt.plot(N_list, value_BTM_list, label = u"二叉树模型的结果", lw = 2.5)
plt.plot(N_list, value_BSM_list, label = u"BSM模型的结果", lw = 2.5)
plt.xlabel(u"步数", fontsize = 13)
plt.xticks(fontsize = 13)
plt.xlim(0, 150) # 设置x轴的刻度为0~150
plt.ylabel(u"期权价值", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"二叉树模型与BSM模型之间的关系图", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 可看出：当二叉树模型步数不断增加时，期权价值走势呈现锯齿形并且二叉树模型的结果收敛于BSM模型的结果，同时二叉树模型的结果围绕着BSM模型的结果上下波动
# 当步数达到80时，二叉树模型的结果与BSM模型的结果几乎无差异，步数趋于无穷大时，通过二叉树模型可以推导出BSM模型



# 11.5 美式期权定价
# 11.5.3 运用矩阵计算
# 针对计算美式看涨期权价值的Python自定义函数
def American_call(S, K, sigma, r, T, N):
    """
    定义一个运用N步二叉树模型计算美式看涨期权价值的函数
    S：代表基础资产当前的价格
    K：代表期权的行权价格
    sigma：代表基础资产收益率的波动率（年化）
    r：代表连续复利的无风险收益率
    T：代表期权的期限（年）
    N：代表二叉树模型的步数
    """
    # 为了便于理解代码编写逻辑，分为以下3个步骤
    # 第1步时计算相关系数
    t = T / N # 计算每一步步长期限（年）
    u = np.exp(sigma * np.sqrt(t)) # 计算基础资产价格上涨时的比例
    d = 1 / u # 计算基础资产价格下跌时的比例
    p = (np.exp(r * t) - d) / (u - d) # 计算基础资产价格上涨的概率
    call_matrix = np.zeros((N+1, N+1)) # 创建N+1行、N+1列的零矩阵，用于后续存放每个节点的期权价值

    # 第2步是计算到期日节点的基础资产价格与期权价值
    N_list = np.arange(0, N+1) # 创建从0到N的自然数数列（数组形式）
    S_end = S * pow(u, N-N_list) * pow(d, N_list) # 计算期权到期时节点的基础资产价格。按照节点从上往下排序，参见式（11-38）
    call_matrix[:, -1] = np.maximum(S_end - K, 0) # 计算期权到期时节点的看涨期权价值（按照节点从上往下排序）

    # 第3步是计算期权非到期日节点的基础资产价格与期权价值
    i_list = list(range(0, N)) # 创建从0到N-1的自然数数列（列表格式）
    i_list.reverse() # 将列表的元素由大到小重新排列（从N-1到0）
    for i in i_list:
        j_list = np.arange(i+1) # 创建从0到i的自然数数列（数组格式）
        Si = S * pow(u, i-j_list) * pow(d, j_list) # 计算在iΔt时刻各节点上的基础资产价格（按照节点从上往下排序）
        call_strike = np.maximum(Si - K, 0) # 计算提前行权时的期权收益
        call_nostrike = (p * call_matrix[: i+1, i+1] + (1 - p) * call_matrix[1: i+2, i+1]) * np.exp(-r * t) # 计算不提前行权时的期权价值
        call_matrix[:i+1, i] = np.maximum(call_strike, call_nostrike) # 取提前行权时的期权收益与不提前行权时的期权价值中的最大值
        call_begin = call_matrix[0, 0] # 期权初始价值
    return call_begin

# 针对计算美式看跌期权价值的Python自定义函数
def American_put(S, K, sigma, r, T, N):
    '''定义一个运用N步二叉树模型计算美式看跌期权价值的函数
    S: 代表基础资产当前的价格。
    K: 代表期权的行权价格。
    sigma: 代表基础资产收益率的波动率（年化）。
    r: 代表连续复利的无风险收益率。
    T: 代表期权的期限（年）。
    N: 代表二叉树模型的步数'''
    # 第1步计算相关参数
    t = T / N
    u = np.exp(sigma * np.sqrt(t))
    d = 1 / u
    p = (np.exp(r * t) - d) / (u - d)
    put_matrix = np.zeros((N + 1, N + 1))
    # 第2步计算期权到期日节点的基础资产价格与期权价值
    N_list = np.arange(0, N + 1)  # 创建从0到N的自然数数列（数组格式）
    S_end = S * u ** (N - N_list) * d ** (N_list)  # 计算期权到期时节点的基础资产价格。按照节点从上往下排序
    put_matrix[:, -1] = np.maximum(K - S_end, 0)  # 计算期权到期时节点的看跌期权价值。按照节点从上往下排序
    # 第3步计算期权非到期日节点的基础资产价格与期权价值
    i_list = list(range(0, N))  # 创建从0到N-1的自然数数列（列表格式）
    i_list.reverse()  # 将列表的元素由大到小重新排序（从N-1到0）
    for i in i_list:
        j_list = np.arange(i + 1)  # 创建从0到i的自然数数列（数组格式）
        Si = S * u ** (i - j_list) * d ** (j_list)  # 计算在iΔt时刻各节点上的基础资产价格（按照节点从上往下排序）
        put_strike = np.maximum(K - Si, 0)  # 计算提前行权时的期权收益
        put_nostrike = (p * put_matrix[:i + 1, i + 1] + (1 - p) * put_matrix[1:i + 2, i + 1]) * np.exp(
            -r * t)  # 计算不提前行权时的期权收益
        put_matrix[:i + 1, i] = np.maximum(put_strike, put_nostrike)  # 取提前行权时的期权收益与不提前行权时的期权收益中的最大值
    put_begin = put_matrix[0, 0]  # 期权初始价值
    return put_begin

# 计算中国银行A股美式看跌期权的价值；并且将步数依次增加至12步（步长为月）、52步（步长为周）和252步（步长为交易日），分别计算该期权的价值
# 第1步：从外部导入2017年至2020年2月7日中国银行A股日收盘价数据，计算股票年化波动率
Price_BOC = pd.read_excel("D:/Python/中国银行A股日收盘价数据（2017年至2020年2月7日）.xls", sheet_name = "Sheet1", header = 0,
                          index_col = 0) # 导入中国银行A股日收盘价的数据

R_BOC = np.log(Price_BOC / Price_BOC.shift(1))  # 计算中国银行A股每日收益率

Sigma_BOC = np.sqrt(252) * np.std(R_BOC) # 计算中国银行股票年化波动率
Sigma_BOC = float(Sigma_BOC) # 转换为浮点型数据
print("中国银行A股年化波动率", round(Sigma_BOC, 4))

# 第2步：运用两步二叉树模型计算美式看跌期权的价值
S_BOC = 3.5 # 中国银行A股2020年2月10日收盘价
K_BOC = 3.8 # 美式看跌期权的行权价格
T_BOC = 1 # 期权期限
r_Feb10 = 0.02 # 2020年2月10日无风险收益率
N_2 = 2 # 步数是2

Put_2step = American_put(S = S_BOC, K = K_BOC, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC, N = N_2) # 运用两步二叉树模型
print("运用两步二叉树模型计算中国银行A股美式看跌期权价值", round(Put_2step, 4))

# 第3步：将二叉树模型的步数依次增加至12步、52步和252步，分别计算该期权的价值
N_12 = 12 # 步数是12
N_52 = 52 # 步数是52
N_252 = 252 # 步数是252

Put_12step = American_put(S = S_BOC, K = K_BOC, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC, N = N_12)
Put_52step = American_put(S = S_BOC, K = K_BOC, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC,N = N_52)
Put_252step = American_put(S = S_BOC, K = K_BOC, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC, N = N_252)
print("运用12步二叉树模型计算中国银行A股美式看跌期权价值", round(Put_12step, 4))
print("运用52步二叉树模型计算中国银行A股美式看跌期权价值", round(Put_52step, 4))
print("运用252步二叉树模型计算中国银行A股美式看跌期权价值", round(Put_252step, 4))

# 11.5.4 美式期权与欧式期权的关系
# 沿用前例信息，在2020年2月10日，期权市场上市了以中国银行A股股票作为基础资产并且期限为1年的欧式期权、美式期权共计12只（见P436）。
# 当天中国银行A股收盘价是3.5元/股，股票收益率的年化波动率是16.76%，无风险收益率是2%
# 通过二叉树模型并借助Python计算以上12只期权的价值，同时设定的步数为252步（以交易日作为步长）
# 第1步：通过自定义函数BTM_Nstep以及American_call，依次计算欧式看涨期权、美式看涨期权的价值
K1 = 3.0 # 行权价格为3元
K2 = 3.5 # 行权价格为3.5元
K3 = 4.0 # 行权价格为4元

Euro_call_K1 = BTM_Nstep(S = S_BOC, K = K1, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC, N = N_252,
                         types = "call") # 计算行权价格为3元的欧式看涨期权价值
Amer_call_K1 = American_call(S = S_BOC, K = K1, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC, N = N_252) # 计算行权价格为3元的美式看涨期权价值
Euro_call_K2 = BTM_Nstep(S = S_BOC, K = K2, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC, N = N_252,
                         types = "call") # 计算行权价格为3.5元的欧式看涨期权价值
Amer_call_K2 = American_call(S = S_BOC, K = K2, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC, N = N_252) # 计算行权价格为3.5元的美式看涨期权价值
Euro_call_K3 = BTM_Nstep(S = S_BOC, K = K3, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC, N = N_252,
                         types = "call") # 计算行权价格为4元的欧式看涨期权价值
Amer_call_K3 = American_call(S = S_BOC, K = K3, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC, N = N_252) # 计算行权价格为4元的美式看涨期权价值
print("行权价格为3元的欧式看涨期权价值", Euro_call_K1)
print("行权价格为3元的美式看涨期权价值", Amer_call_K1)
print("行权价格为3.5元的欧式看涨期权价值", Euro_call_K2)
print("行权价格为3.5元的美式看涨期权价值", Amer_call_K2)
print("行权价格为4元的欧式看涨期权价值", Euro_call_K3)
print("行权价格为4元的美式看涨期权价值", Amer_call_K3)
"""
1.可看出：相同基础资产、相同行权价格和相同期限的欧式看涨期权与美式看涨期权，在期权价值上几乎是完全相同的，
差异仅仅出现在小数点后的第14位至第17位
2.当期权的基础资产不产生期间收益（比如无股息股票）时，提前行使该美式看涨期权不是最优的选择，此时美式看涨期权就退化为欧式看涨期权）
"""

# 第2步：通过自定义函数BTM_Nstep以及American_put，依次计算欧式看跌期权、美式看跌期权的价值
Euro_put_K1 = BTM_Nstep(S = S_BOC, K = K1, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC, N = N_252,
                        types = "put") # 计算行权价格为3元的欧式看跌期权价值
Amer_put_K1=American_put(S = S_BOC, K = K1, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC,
                         N = N_252) # 计算行权价格为3元的美式看跌期权价值
Euro_put_K2 = BTM_Nstep(S = S_BOC, K = K2, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC, N = N_252,
                        types = "put") # 计算行权价格为3.5元的欧式看跌期权价值
Amer_put_K2=American_put(S = S_BOC, K = K2, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC,
                         N = N_252) # 计算行权价格为3.5元的美式看跌期权价值
Euro_put_K3 = BTM_Nstep(S = S_BOC, K = K3, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC, N = N_252,
                        types = "put") # 计算行权价格为4元的欧式看跌期权价值
Amer_put_K3=American_put(S = S_BOC, K = K3, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC,
                         N = N_252) # 计算行权价格为4元的美式看跌期权价值
print("行权价格为3元的欧式看跌期权价值", Euro_put_K1)
print("行权价格为3元的美式看跌期权价值", Amer_put_K1)
print("行权价格为3.5元的欧式看跌期权价值", Euro_put_K2)
print("行权价格为3.5元的美式看跌期权价值", Amer_put_K2)
print("行权价格为4元的欧式看跌期权价值", Euro_put_K3)
print("行权价格为4元的美式看跌期权价值", Amer_put_K3)
# 可看出：美式看跌期权价值要明显高于相同行权价格的欧式看跌期权价值。这意味着美式看跌期权在存续期内都存在被提前行权的可能性

# 第3步：以行权价格为3元的欧式看跌期权、美式看跌期权作为分析对象，同时对于中国银行A股当前股价取值是处于[1.0, 5.0]区间的等差数列，计算对应于不同股价的期权价值并且进行可视化
S_BOC_list = np.linspace(1.0, 5.0, 200) # 设定中国银行A股股价的等差数列（数组格式）

Euro_put_list = np.zeros_like(S_BOC_list) # 创建与股价等差数列形状相同的数组，用于存放欧式看跌期权价值
Amer_put_list=np.zeros_like(S_BOC_list) # 创建与股价等差数列形状相同的数组，用于存放美式看跌期权价值

for i in range(len(S_BOC_list)): # 通过for语句计算对应股价的期权价值
    Euro_put_list[i] = BTM_Nstep(S = S_BOC_list[i], K = K1, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC, N = N_252,
                                 types = "put") # 计算对应于不同股价的欧式看跌期权价值
    Amer_put_list[i] = American_put(S = S_BOC_list[i], K = K1, sigma = Sigma_BOC, r = r_Feb10, T = T_BOC,
                                    N = N_252) # 计算对应于不同股价的美式看跌期权价值

plt.figure(figsize = (9, 6))
plt.plot(S_BOC_list, Euro_put_list, "b-", label = u"欧式看跌期权", lw = 2.5)
plt.plot(S_BOC_list, Amer_put_list, "r-", label = u"美式看跌期权", lw = 2.5)
plt.xlabel(u"中国银行A股股价", fontsize = 13)
plt.xticks(fontsize = 13)
plt.ylabel(u"期权价值", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"欧式看跌期权与美式看跌期权的价值关系图", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()
# 可看出：看跌期权的实值程度越深（股价比行权价格3元越低），美式看跌期权与欧式看跌期权的价值差异就越大，说明美式看跌期权越可能被提前行权；
# 相反，看跌期权的虚值程度越深（股价比行权价格3元越高），美式看跌期权的价值就越趋于欧式看跌期权的价值，期权被提前行权的可能性就越小

#第4步：考察期权价值与内在价值的关系
Intrinsic_value = np.maximum(K1 - S_BOC_list, 0) # 看跌期权的内在价值

plt.figure(figsize = (9, 6))
plt.plot(S_BOC_list, Amer_put_list, "r-", label = u"美式看跌期权价值", lw = 2.5)
plt.plot(S_BOC_list, Intrinsic_value, "g--", label = u"期权内在价值", lw = 2.5)
plt.xlabel(u"中国银行A股股价", fontsize = 13)
plt.xticks(fontsize = 13)
plt.ylabel(u"期权价值", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"美式看跌期权价值与期权内在价值的关系图", fontsize = 13)
plt.legend(fontsize = 13, loc = 9)
plt.grid()
plt.show()
# 可看出：对于美式看跌期权而言，随着期权实值程度加深，期权价值逐渐收敛于期权内在价值，并且当股价小于2.5元/股时，期权价值与期权内在价值几乎无差异；
# 随着期权虚值程度加深，期权价值也不断收敛于期权内在价值，并且当股价高于4元/股时，两者之间也几乎无差异

plt.figure(figsize = (9, 6))
plt.plot(S_BOC_list, Euro_put_list, "r-", label = u"欧式看跌期权价值", lw = 2.5)
plt.plot(S_BOC_list, Intrinsic_value, "g--", label = u"期权内在价值", lw = 2.5)
plt.xlabel(u"中国银行A股股价", fontsize = 13)
plt.xticks(fontsize = 13)
plt.ylabel(u"期权价值", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"欧式看跌期权价值与期权内在价值的关系图", fontsize = 13)
plt.legend(fontsize = 13, loc = 9)
plt.grid()
plt.show()
# 可看出：此图与前图有类似的规律，但随着期权实值程度加深，尤其当股价小于2.5元/股时，欧式看跌期权价值低于期权内在价值，
# 即对于欧式看跌期权，随着期权实值程度加深，期权时间价值将由正转负
