# 第九章
# 互换市场的概况
# 利率互换市场
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["FangSong"]
mpl.rcParams["axes.unicode_minus"] = False
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# 从外部导入数据
IRS_data = pd.read_excel("C:/Users/InsaneHe/desktop/Python/利率互换交易规模.xlsx", sheet_name = "Sheet1", header = 0,
                         index_col = 0)

# 获取数据框关于参考利率类型的利率名称
name = IRS_data.index

# 将数据框涉及名义本金的数值转为一维数据
volume = (np.array(IRS_data)).ravel()

plt.figure(figsize = (9, 7))
plt.pie(x = volume, labels = name, textprops = {"fontsize": 13})
plt.axis("equal")# 使饼图是一个圆形
plt.show()
# 可看出：参考利率以7天期银行间回购定盘利率（FR007）和3个月期上海银行间同业拆放利率（Shibor3M）为主导



# 货币互换市场
# 不同交换币种名称
currency = ["美元兑人民币", "非美元外币与人民币"]

# 对应不同交换币种的人民币外汇货币掉期成交金额（亿元）
volume1 = [158.45, 10.67]

# 不同期限名称
tenor = ["不超过1年", "超过1年"]

# 对应不同期限的人民币外汇货币掉期成交金额（亿元）
volume2 = [141.29, 27.83]

plt.figure(figsize=(11, 7))
plt.subplot(1, 2, 1)# 第1张子图
plt.pie(x = volume1, labels = currency, textprops = {"fontsize": 13})
plt.axis("equal")# 使饼图是一个圆形
plt.title(u"不同交换币种", fontsize = 14)
plt.subplot(1, 2, 2)# 第2张子图
plt.pie(x = volume2, labels = tenor, textprops = {"fontsize": 13})
plt.axis("equal")
plt.title(u"不同期限", fontsize = 14)
plt.show()
# 可看出：美元与人民币作为交换币种以及期限不超过1年的货币掉期合约均是市场交易的主力合约



# 信用违约互换市场
# 从外部导入数据
CRM_data = pd.read_excel("C:/Users/InsaneHe/desktop/Python/未到期信用风险缓释工具合约面值（2020年年末）.xlsx",
                         sheet_name = "Sheet1", header = 0, index_col = 0)

# 获取数据框中关于合约创设机构类型
type_CRM = CRM_data.index

# 将数据框涉及合约面值的数值转为一维数组
par_CRM = (np.array(CRM_data)).ravel()

plt.figure(figsize=(9, 7))
plt.pie(x = par_CRM, labels = type_CRM, textprops = {"fontsize": 13})
plt.axis("equal")# 使饼图是一个圆形
plt.show()
# 可看出：商业银行作为创设机构发行的信用风险缓释工具合约面值规模超过了一半，其次是中债信用增进公司，证券公司的合约面值最少



# 利率互换
# 利率互换的期间现金流
# 通过Python自定义一个计算利率互换合约存续期内各交易方支付利息净额的函数
def IRS_cashflow(R_flt, R_fix, L, m, position):
    """
    定义一个计算利率互换合约存续期内每期支付利息净额的函数
    R_flt：代表利率互换的每期浮动利率，以数组格式输入
    R_fix：代表利率互换的固定利率
    L：代表利率互换的本金
    m：代表利率互换存续期内每年交换利息的频次
    position：代表头寸方向，position = "long"代表多头（支付固定利息，收取浮动利息），输入position = "short"代表空头（支付浮动利息，收取固定利息）
    """
    if position == "long":# 当交易方是合约多头
        cashflow = (R_flt - R_fix) * L / m# 计算利率互换多头每期的净现金流
    else:# 当交易方是合约空头
        cashflow = (R_fix - R_flt) * L / m# 计算利率互换空头每期的净现金流
    return cashflow

# 计算例9-1中在利率互换合约存续期内A银行和B银行每期利息支付净额
# 以数组格式输入Shibor
rate_float = np.array([0.031970, 0.032000, 0.029823, 0.030771, 0.044510, 0.047093, 0.043040, 0.032750, 0.029630,
                       0.015660])

# 利率互换的固定利率
rate_fixed = 0.037

# 利率互换的本金
par = 1e8

# 利率互换每年交换利息的频次
M = 2

# A银行（多头）的每期利息支付净额
Netpay_A = IRS_cashflow(R_flt = rate_float, R_fix = rate_fixed, L = par, m = M, position = "long")
print(Netpay_A)

# B银行（空头）的每期利息支付净额
Netpay_B = IRS_cashflow(R_flt = rate_float, R_fix = rate_fixed, L = par, m = M, position = "short")
print(Netpay_B)

# 计算A银行利息支付净额的合计数
Totalpay_A = np.sum(Netpay_A)

# 计算B银行利息支付净额的合计数
Totalpay_B = np.sum(Netpay_B)

print("利率互换合约存续期内A银行利息支付净额的合计数", round(Totalpay_A, 2))
print("利率互换合约存续期内B银行利息支付净额的合计数", round(Totalpay_B, 2))



# 利率互换的等价性
# 利率互换的多头：浮动利率债券多头+固定利率债券空头；利率互换的空头：浮动利率债券空头+固定利率债券多头
# 利率互换合约的初始价值为0（满足该关系的固定利率为互换利率）



# 互换利率的计算
# 通过Python自定义一个计算互换利率的函数
def swap_rate(m, y, T):
    """
    定义一个计算互换利率的函数
    m：代表利率互换合约存续期内每年交换利息的频次
    y：代表合约初始日对应于每期利息交换期限，连续复利的零息利率（贴现利率），用数组格式输入
    T：代表利率互换的期限（年）
    """
    n_list = np.arange(1, m * T + 1)# 创建1~mT的整数数组
    t = n_list / m# 计算合约初始日距离每期利息交换日的期限数组
    q = np.exp(-y * t)# 计算针对不同期限的贴现因子（数组格式）
    rate = m * (1 - q[-1]) / np.sum(q)# 计算互换利率
    return rate

# 计算互换利率的案例
# C银行准备与交易对手D银行开展一笔利率互换业务，该利率互换合约的期限是3年，合约初始日是2020年7月1日，到期日是2023年7月1日，
# 利率互换合约的本金是1亿元。同时，在合约中约定，双方每6个月交换一次利息。其中，C银行支付固定利息并且收取按照6个月期Shibor计算的浮动利息，
# D银行收取固定利息并且支付浮动利息。C银行和D银行都需要计算该利率互换合约的互换利率
# 利率互换合约每年交换利息的频次
freq = 2

# 利率互换合约的期限
tenor = 3

# 输入对应于不同利息交换期限的零息利率
r_list = np.array([0.020579, 0.021276, 0.022080, 0.022853, 0.023527, 0.024036])

# 计算互换利率
R_July1 = swap_rate(m = freq, y = r_list, T = tenor)
print("2020年7月1日利率互换合约的互换利率", round(R_July1, 4))



# 利率互换的定价
# 通过Python自定义一个函数计算利率互换合约的价值
def swap_value(R_fix, R_flt, t, y, m, L, position):
    """
    定义一个计算合约存续期内利率互换合约价值的函数
    R_fix：代表利率互换合约的固定利率（互换利率）
    R_flt：代表距离合约定价日最近的下一期利息交换的浮动利率
    t：代表合约定价日距离每期利息交换日的期限（年），用数组格式输入
    y：代表期限为t并且连续复利的零息利率（贴现利率），用数组格式输入
    m：代表利率互换合约每年交换利息的频次
    L：代表利率互换合约的本金
    position：代表头寸方向，输入position="long"代表多头（支付固定利息，收取浮动利息），输入position="short"代表空头（支付浮动利息，收取固定利息）
    """
    from numpy import exp# 从NumPy模块导入exp函数
    B_fix = (R_fix * sum(exp(-y * t)) / m + exp(-y[-1] * t[-1])) * L# 计算固定利率债券价值
    B_flt = (R_flt / m + 1) * L * exp(-y[0] * t[0])# 计算浮动利率债券价值
    if position == "long":# 针对合约多头
        value = B_flt - B_fix# 计算互换利率合约多头的价值
    else:# 针对合约空头
        value = B_fix - B_flt# 计算互换利率合约空头的价值
    return value

# 沿用前例，C银行在2020年7月1日与D银行开展了本金为1亿元、期限为3年的利率互换合约，每6个月交换一次利息，C银行支付固定利息，
# D银行支付基于6个月期Shibor的浮动利息，固定利率设定为2.41%，2020年7月1日的6个月期Shibor等于2.178%。依次计算在这两个交易日针对C银行和D银行而言，该利率互换合约的价值
# 第1步：通过表9-10（P313）中已知的零息利率，运用3阶样条曲线插值法计算1.5年期和2.5年期的零息利率
# 导入SciPy的模块interpolate
import scipy.interpolate as si

# 输入表9-10中的相关期限
T = np.array([1/12, 2/12, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0])

# 2020年7月10日已知的零息利率
R_July10 = np.array([0.017219, 0.017526, 0.021012, 0.021100, 0.021764, 0.022165, 0.025040, 0.026894])

# 2020年7月20日已知的零息利率
R_July20 = np.array([0.016730, 0.018373, 0.019934, 0.020439, 0.021621, 0.022540, 0.024251, 0.025256])

# 运用2020年7月10日的零息利率数据和3阶样条曲线插值法构建插值函数
func_July10 = si.interp1d(x = T, y = R_July10, kind = "cubic")

# 运用2020年7月20日的零息利率数据和3阶样条曲线插值法构建插值函数
func_July20 = si.interp1d(x = T, y = R_July20, kind = "cubic")

# 输入包含1.5年和2.5年的新期限数组
T_new = np.array([1/12, 2/12, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])

# 用插值法计算2020年7月10日的新零息利率
R_new_July10 = func_July10(T_new)

# 查看输出结果
print(R_new_July10)

# 用插值法计算2020年7月20日的新零息利率
R_new_July20 = func_July20(T_new)

# 查看输出结果
print(R_new_July20)

# 第2步：计算合约定价日距离每期利息交换日的期限（距离合约定价日最近的下一期利息交换日为2021年1月1日）
# 导入datetime模块
import datetime as dt

# 输入2020年7月10日
T1 = dt.datetime(2020, 7, 10)

# 输入2020年7月20日
T2 = dt.datetime(2020, 7, 20)

# 输入下一期利息交换日
T3 = dt.datetime(2021, 1, 1)

# 计算2020年7月10日至2021年1月1日的期限（年）
tenor1 = (T3 - T1).days / 365

# 计算2020年7月20日至2021年1月1日的期限（年）
tenor2 = (T3 - T2).days / 365

# 利率互换的总期限
T = 3

# 每年交换利息的频次
M = 2

# 创建存放2020年7月10日距离每期利息交换日期限的初始数组
T_list1 = np.arange(T * M) / M

# 计算相关期限
T_list1 = T_list1 + tenor1
print(T_list1)

# 创建存放2020年7月20日距离每期利息交换日期限的初始数组
T_list2 = np.arange(T * M) / M

# 计算相关期限
T_list2 = T_list2 + tenor2
print(T_list2)
# 针对以上输出的两个期限数组，第1个元素代表合约定价日距离第1期利息交换日的期限长度，第2个元素代表合约定价日距离第2期利息交换日的期限长度

# 第3步：运用自定义函数swap_value，计算两个不同交易日的合约价值
# 创建存放2020年7月10日对应每期利息交换期限的零息利率初始数组
yield_July10 = np.zeros_like(T_list1)

# 存放2020年7月10日6个月期零息利率
yield_July10[0] = R_new_July10[3]

# 存放2020年7月10日1年期、1.5年期、2年期、2.5年期和3年期零息利率
yield_July10[1:] = R_new_July10[5:]

# 创建存放2020年7月20日对应每期利息交换期限的零息利率初始数组
yield_July20 = np.zeros_like(T_list2)

# 存放2020年7月10日6个月期零息利率
yield_July20[0] = R_new_July20[3]

# 存放2020年7月10日1年期、1.5年期、2年期、2.5年期和3年期零息利率
yield_July20[1:] = R_new_July20[5:]

# 互换利率（固定利率）
rate_fix = 0.0241

# 第1期利息交换的浮动利率
rate_float = 0.02178

# 利率互换的名义本金
par = 1e8

# 2020年7月10日对于C银行的利率互换合约价值
value_July10_long = swap_value(R_fix = rate_fix, R_flt = rate_float, t = T_list1, y = yield_July10, m = M, L = par,
                               position = "long")

# 2020年7月10日对于D银行的利率互换合约价值
value_July10_short = swap_value(R_fix = rate_fix, R_flt = rate_float, t = T_list1, y = yield_July10, m = M, L = par,
                                position = "short")

print("2020年7月10日C银行（多头）的利率互换合约价值", round(value_July10_long, 2))
print("2020年7月10日D银行（空头）的利率互换合约价值", round(value_July10_short, 2))

# 2020年7月20日对于C银行的利率互换合约价值
value_July20_long = swap_value(R_fix = rate_fix, R_flt = rate_float, t = T_list2, y = yield_July20, m = M, L = par,
                               position = "long")

# 2020年7月20日对于D银行的利率互换合约价值
value_July20_short = swap_value(R_fix = rate_fix, R_flt = rate_float, t = T_list2, y = yield_July20, m = M, L = par,
                                position = "short")

print("2020年7月20日C银行（多头）的利率互换合约价值", round(value_July20_long, 2))
print("2020年7月20日D银行（空头）的利率互换合约价值", round(value_July20_short, 2))
# 可看出：在不同的交易日由于零息利率的变动以及合约定价日距离利息交换日期限长度的变化，导致合约价值发生变动。
# 此外，合约空头的价值恰好是合约多头的价值的相反数，说明利率互换合约实质是零和博弈



# 货币互换
# 对利率互换合约做一些改造，在合约初始日以及到期日都进行本金的交换，且用于交换的本金按照两种不同的币种计价，此时即为货币互换

# 货币互换根据合约期间交换利率的特征可划分为三大类
"""
货币互换的类型及特征：
1.双固定利率货币互换：将一种货币下的固定利息及本金与另一种货币下的固定利息及本金进行交换
2.固定对浮动货币互换：将一种货币下的固定利息及本金与另一种货币下的浮动利息及本金进行交换（也称为交叉货币利率互换）
3.双浮动利率货币互换：将一种货币下的浮动利息及本金与另一种货币下的浮动利息及本金进行交换
"""

# 双固定利率货币互换的期间现金流
# 通过Python自定义一个计算双固定利率货币互换在存续期间每期现金流的函数
def CCS_fixed_cashflow(La, Lb, Ra_fix, Rb_fix, m, T, trader, par):
    """
    定义一个计算双固定利率货币互换在存续期间每期现金流的函数
    合约的交易双方分别用A交易方和B交易方表示
    La：代表在合约初始日A交易方支付的一种货币本金（合约到期日A交易方收回的货币本金）
    Lb：代表在合约初始日B交易方支付的另一种货币本金（合约到期日B交易方收回的货币本金）
    Ra_fix：代表基于本金La的固定利率
    Rb_fix：代表基于本金Lb的固定利率
    m：代表每年交换利息的频率
    T：代表货币互换合约的期限（年）
    trader：代表合约的交易方，输入trader="A"表示计算A交易方发生的期间现金流，输入trader="B"表示计算B交易方发生的期间现金流
    par：代表计算现金流所依据的本金，输入par="La"表示计算的现金流基于本金La，输入其他则表示计算的现金流就本金Lb
    """
    cashflow = np.zeros(m * T + 1)# 创建存放每期现金流的初始数组
    if par == "La":# 依据本金La计算现金流
        cashflow[0] = -La# 计算A交易方第1期现金流
        cashflow[1: -1] = Ra_fix * La / m# 计算A交易方第2期至倒数第2期的现金流
        cashflow[-1] = (Ra_fix / m + 1) * La# 计算A交易方最后一期的现金流
        if trader == "A":# 针对A交易方
            return cashflow# 以A货币计
        else:
            return -cashflow# 以A货币计
    else:# 依据本金Lb计算现金流
        cashflow[0] = Lb# 计算A交易方第1期的现金流
        cashflow[1: -1] = -Rb_fix * Lb / m# 计算A交易方第2期至倒数第2期的现金流
        cashflow[-1] = -(Rb_fix / m + 1) * Lb# 计算A交易方最后一期的现金流
        if trader == "A":# 针对A交易方
            return cashflow# 以B货币计
        else:# 针对B交易方
            return -cashflow# 以B货币计

# 结合案例的Python编程
# 运用Python自定义函数计算例9-4中的每期现金流数据（p316）
# 第1步：在Python中输入相关的合约参数
# E银行在货币互换合约初始日支付的人民币本金
par_RMB = 6.4e8

# F银行在货币互换合约初始日支付的美元本金
par_USD = 1e8

# 人民币本金的利率
rate_RMB = 0.02

# 美元本金的利率
rate_USD = 0.01

# 货币互换合约每年交换利息的频次
M = 2

# 货币互换合约的期限（年）
tenor = 5

# 第2步：运用自定义函数CCS_fixed_cashflow，计算针对不同交易方并依据不同币种本金得到的每期现金流
# 计算E银行基于人民币本金的每期现金流
cashflow_Ebank_RMB = CCS_fixed_cashflow(La = par_RMB, Lb = par_USD, Ra_fix = rate_RMB, Rb_fix = rate_USD, m = M,
                                        T = tenor, trader = "A", par = "La")

# 计算E银行基于美元本金的每期现金流
cashflow_Ebank_USD = CCS_fixed_cashflow(La = par_RMB, Lb = par_USD, Ra_fix = rate_RMB, Rb_fix = rate_USD, m = M,
                                        T = tenor, trader = "A", par = "Lb")

print("E银行基于人民币本金的每期现金流（人民币）\n", cashflow_Ebank_RMB)
print("E银行基于美元本金的每期现金流（美元）\n", cashflow_Ebank_USD)

# 计算F银行基于人民币本金的每期现金流
cashflow_Fbank_RMB = CCS_fixed_cashflow(La = par_RMB, Lb = par_USD, Ra_fix = rate_RMB, Rb_fix = rate_USD, m = M,
                                        T = tenor, trader = "B", par = "La")

# 计算F银行基于美元本金的每期现金流
cashflow_Fbank_USD = CCS_fixed_cashflow(La = par_RMB, Lb = par_USD, Ra_fix = rate_RMB, Rb_fix = rate_USD, m = M,
                                        T = tenor, trader = "B", par = "Lb")

print("F银行基于人民币本金的每期现金流（人民币）\n", cashflow_Fbank_RMB)
print("F银行基于美元本金的每期现金流（美元）\n", cashflow_Fbank_USD)



# 固定对浮动货币互换的期间现金流
# 假定本金La支付固定利息而Lb支付浮动利息
# Python自定义函数
# 基于例9-14的现金流表达式（p319），通过Python自定义一个计算固定对浮动货币互换在存续期间每期现金流的函数
def CCS_fixflt_cashflow(La, Lb, Ra_fix, Rb_flt, m, T, trader, par):
    """
    定义一个计算固定对浮动货币互换在存续期间每期现金流的函数
    合约的交易双方依然分别用A交易方和B交易方表示
    La：代表在合约初始日A交易方支付的一种货币本金（合约到期日A交易方收回的货币本金）
    Lb：代表在合约初始日B交易方支付的一种货币本金（合约到期日B交易方收回的货币本金）
    Ra_fix：代表基于本金La的固定利率
    Rb_flt：代表基于本金Lb的浮动利率，并且以数组格式输入
    m：代表每年交换利息的频次
    T：代表货币互换合约的期限（年）
    trader：代表合约的交易方，输入trader="A"表示计算A交易方发生的期间现金流，输入其他则表示计算B交易方发生的期间现金流
    par：代表计算现金流所依据的本金，输入par="La"表示计算的现金流基于本金La，输入其他则表示计算的现金流基于本金Lb
    """
    cashflow = np.zeros(m * T + 1)# 创建存放每期现金流的初始数组
    if par == "La":# 依据本金La计算现金流
        cashflow[0] = -La# A交易方第1期交换的现金流
        cashflow[1: -1] = Ra_fix * La / m# A交易方第2期至倒数第2期的现金流
        cashflow[-1] = (Ra_fix / m + 1) * La# A交易方最后一期的现金流
        if trader == "A":# 针对A交易方
            return cashflow
        else:# 针对B交易方
            return -cashflow
    else:# 依据本金Lb计算现金流
        cashflow[0] = Lb# A交易方第1期交换的现金流
        cashflow[1: -1] = -Rb_flt[: -1] * Lb / m# A交易方第2期至倒数第2期交换的现金流
        cashflow[-1] = -(Rb_flt[-1] / m + 1) * Lb# A交易方最后一期交换的现金流
        if trader == "A":# 针对A交易方
            return cashflow
        else:# 针对B交易方
            return -cashflow



# 双浮动利率货币互换的期间现金流
# Python自定义函数
# 基于表9-15的现金流表达式（p321），通过Python自定义一个计算双浮动利率货币互换在存续期间每期现金流的函数
def CCS_float_cashflow(La, Lb, Ra_flt, Rb_flt, m, T, trader, par):
    """
    定义一个计算双浮动利率货币互换在存续期间每期现金流的函数
    合约的交易双方分别用A交易方和B交易方表示
    La：代表在合约初始日A交易方支付的一种货币本金（合约到期日收到的货币本金）
    Lb：代表在合约初始日B交易方支付的一种货币本金（合约到期日收到的货币本金）
    Ra_flt：代表基于本金La的浮动利率，以数组格式输入
    Rb_flt：代表基于本金Lb的浮动利率，以数组格式输入
    m：代表每年交换利息的频次
    T：代表货币互换合约的期限（年）
    trader：代表合约的交易方，输入trader="A"就表示计算A交易方发生的期间现金流，输入其他则表示计算B交易方发生的期间现金流
    par：代表计算现金流所依据的本金，输入par="La"表示计算的现金流基于本金La，输入其他则表示计算的现金流基于本金Lb
    """
    cashflow = np.zeros(m * T + 1)# 创建存放每期现金流的初始数组
    if par == "La":# 依据本金La计算现金流
        cashflow[0] = -La# A交易方第1期的现金流
        cashflow[1: -1] = Ra_flt[:-1] * La / m# A交易方第2期至倒数第2期的现金流
        cashflow[-1] = (Ra_flt[-1] / m + 1) * La# A交易方最后一期的现金流
        if trader == "A":# 针对A交易方
            return cashflow
        else:# 针对B交易方
            return -cashflow
    else:# 依据本金Lb计算现金流
        cashflow[0] = Lb# A交易方第1期的现金流
        cashflow[1: -1] = Rb_flt[: -1] * Lb / m# A交易方第2期至倒数第2期的现金流
        cashflow[-1] = (Rb_flt[-1] / m + 1) * Lb# A交易方最后一期的现金流
        if trader == "A":# 针对A交易方
            return cashflow
        else:# 针对B交易方
            return -cashflow

# 结合案例的Python编程
# 2016年12月1日，G银行分别与H银行、I银行达成了两份货币互换合约，一份是固定对浮动货币互换合约的期间现金流，另一份是双浮动利率货币互换
# 第1份货币互换合约是固定对浮动货币互换。合约交易方分别是G银行和H银行，合约期限是3年，
# 在合约初始日，G银行支付给H银行6.9亿元，H银行支付给G银行1亿美元；在合约到期日，G银行收回6.9亿元，H银行收回1亿美元；
# 在合约存续期内，G银行与H银行每半年交换一次利息，其中，针对人民币本金采用3%/年的固定利率，针对美元本金则采用6个月期美元Libor（浮动利率）

# 第2份货币互换合约是浮动对浮动货币互换。合约交易方分别是G银行和I银行，合约期限是4年，
# 在合约初始日，G银行支付给I银行1.8亿元，I银行支付给G银行2亿港元；在合约到期日，G银行收回1.8亿元，I银行收回2亿港元；
# 在合约存续期内，G银行与I银行每年交换一次利息，其中，针对人民币本金采用1年期Shibor（浮动利率），针对港元本金采用1年期港元Hibor（浮动利率）
# 运用前面的Python自定义函数计算这两份货币互换合约在存续期内的现金流
# 第1步：根据案例提供的信息，输入相关的合约参数
# 第1份货币互换合约的人民币本金
par_RMB1 = 6.9e8

# 第1份货币互换合约的美元本金
par_USD = 1e8

# 第2份货币互换合约的人民币本金
par_RMB2 = 1.8e8

# 第2份货币互换合约的港元本金
par_HKD = 2e8

# 第1份货币互换合约每年交换利息的次数
M1 = 2

# 第2份货币互换合约每年交换利息的次数
M2 = 1

# 第1份货币互换合约的期限（年）
T1 = 3

# 第2份货币互换合约的期限（年）
T2 = 4

# 第1份货币互换合约基于人民币本金的固定利率
rate_fix = 0.03

# 第1份货币互换合约基于美元本金的浮动利率
Libor = np.array([0.012910, 0.014224, 0.016743, 0.024744, 0.028946, 0.025166])

# 第2份货币互换合约基于人民币本金的浮动利率
Shibor = np.array([0.031600, 0.046329, 0.035270, 0.031220])

# 第2份货币互换合约基于港元本金的浮动利率
Hibor = np.array([0.013295, 0.015057, 0.026593, 0.023743])

# 第2步：根据自定义函数CCS_fixflt_cashflow以及第1步输入的参数，计算第1份货币互换合约在存续期内不同交易方的现金流
# 第1份货币互换合约在存续期内G银行的人民币现金流
cashflow_Gbank_RMB1 = CCS_fixflt_cashflow(La = par_RMB1, Lb = par_USD, Ra_fix = rate_fix, Rb_flt = Libor, m = M1,
                                          T = T1, trader = "A", par = "La")

# 第1份货币互换合约在存续期G银行的美元现金流
cashflow_Gbank_USD = CCS_fixflt_cashflow(La = par_RMB1, Lb = par_USD, Ra_fix = rate_fix, Rb_flt = Libor, m = M1,
                                         T = T1, trader = "A", par = "Lb")

print("第1份货币互换合约在存续期内G银行的人民币现金流\n", cashflow_Gbank_RMB1)
print("第1份货币互换合约在存续期内G银行的美元现金流\n", cashflow_Gbank_USD)

# 第1份货币互换合约在存续期内H银行的人民币现金流
cashflow_Hbank_RMB = CCS_fixflt_cashflow(La = par_RMB1, Lb = par_USD, Ra_fix = rate_fix, Rb_flt = Libor, m = M1,
                                         T = T1, trader = "B", par = "La")

# 第1份货币互换合约在存续期内H银行的美元现金流
cashflow_Hbank_USD = CCS_fixflt_cashflow(La = par_RMB1, Lb = par_USD, Ra_fix = rate_fix, Rb_flt = Libor, m = M1,
                                         T = T1, trader = "B", par = "Lb")

print("第1份货币互换合约在存续期内H银行的人民币现金流\n", cashflow_Hbank_RMB)
print("第1份货币互换合约在存续期内H银行的美元现金流\n", cashflow_Hbank_USD)

# 第3步：根据前面自定义函数CCS_float_cashflow以及第1步输入的参数，计算第2份货币互换合约在存续期内不同交易方的现金流
# 第2份货币互换合约在存续期内G银行的人民币现金流
cashflow_Gbank_RMB2 = CCS_float_cashflow(La = par_RMB2, Lb = par_HKD, Ra_flt = Shibor, Rb_flt = Hibor, m = M2,
                                         T = T2, trader = "A", par = "La")

# 第2份货币互换合约在存续期内G银行的港元现金流
cashflow_Gbank_HKD = CCS_float_cashflow(La = par_RMB2, Lb = par_HKD, Ra_flt = Shibor, Rb_flt = Hibor, m = M2,
                                        T = T2, trader = "A", par = "Lb")

print("第2份货币互换合约在存续期内G银行的人民币现金流\n", cashflow_Gbank_RMB2)
print("第2份货币互换合约在存续期内G银行的港元现金流\n", cashflow_Gbank_HKD)

# 第2份货币互换合约在存续期内I银行的人民币现金流
cashflow_Ibank_RMB = CCS_float_cashflow(La = par_RMB2, Lb = par_HKD, Ra_flt = Shibor, Rb_flt = Hibor, m = M2,
                                        T = T2, trader = "B", par = "La")

# 第2份货币互换合约在存续期内I银行的港元现金流
cashflow_Ibank_HKD = CCS_float_cashflow(La = par_RMB2, Lb = par_HKD, Ra_flt = Shibor, Rb_flt = Hibor, m = M2,
                                        T = T2, trader = "B", par = "Lb")

print("第2份货币互换合约在存续期内I银行的人民币现金流\n", cashflow_Ibank_RMB)
print("第2份货币互换合约在存续期内I银行的港元现金流\n", cashflow_Ibank_HKD)



# 货币互换的等价性与定价
# 货币互换的现金流等价于包含两只债券的投资组合，且这两只债券的计价货币不同
# 相当于各方开始时买入本国货币债券，卖出他国货币债券
# 在货币互换中，通常A交易方以A货币计价，B交易方以B货币计价，故须考虑汇率
# Python自定义函数
# 结合货币互换定价数学表达式，通过Python自定义一个计算在合约存续期内货币互换合约价值的函数
def CCS_value(types, La, Lb, Ra, Rb, ya, yb, E, m, t, trader):
    """
    定义一个计算在合约存续期内货币互换合约价值的函数，交易双方是A交易方和B交易方，
    同时约定A交易方在合约初始日支付A货币本金，B交易方在合约初始日支付B货币本金
    types：代表货币互换类型，输入types="双固定利率货币互换"表示计算双固定利率货币互换，输入types="双浮动利率货币互换"表示计算双浮动利率货币互换，
    输入其他则表示计算固定对浮动利率货币互换；并约定针对固定对浮动利率货币互换，固定利率针对A货币本金，浮动利率针对B货币本金
    La：代表A货币本金
    Lb：代表B货币本金
    Ra：代表针对A货币本金的利率
    Rb：代表针对B货币本金的利率
    ya：代表在合约定价日针对A货币本金并对应不同期限、连续复利的零息利率，用数组格式输入
    yb：代表在合约定价日针对B货币本金并对应不同期限、连续复利的零息利率，用数组格式输入
    E：代表在合约定价日的即期汇率，标价方式是1单位B货币对应的A货币数量
    m：代表每年交换利息的频次
    t：代表合约定价日距离剩余每期利息交换日的期限长度，用数组格式输入
    trader：代表交易方，输入trader="A"表示A交易方，输入其他则表示B交易方
    """
    from numpy import exp# 从NumPy模块导入exp函数
    if types == "双固定利率货币互换":# 当货币互换类型是双固定利率货币互换时
        Bond_A = (Ra * sum(exp(-ya * t)) / m + exp(-ya[-1] * t[-1])) * La# 计算对应A货币本金的固定利率债券价值
        Bond_B = (Rb * sum(exp(-yb * t)) / m + exp(-yb[-1] * t[-1])) * Lb# 计算对应B货币本金的固定利率债券价值
        if trader == "A":# 针对A交易方
            swap_value = Bond_A - Bond_B * E# 计算货币互换合约的价值（以A货币计价）
        else:# 针对B交易方
            swap_value = Bond_B - Bond_A / E# 计算货币互换合约的价值（以B货币计价）
    elif types == "双浮动利率货币互换":# 当货币互换类型时双浮动利率货币互换时
        Bond_A = (Ra / m + 1) * exp(-ya[0] * t[0]) * La# 计算对应A货币本金的浮动利率债券价值
        Bond_B = (Rb / m + 1) * exp(-yb[0] * t[0]) * Lb# 计算对应B货币本金的浮动利率债券价值
        if trader == "A":
            swap_value = Bond_A - Bond_B * E
        else:
            swap_value = Bond_B - Bond_A / E
    else:# 当货币互换类型时固定对浮动货币互换时
        Bond_A = (Ra * sum(exp(-ya * t)) / m + exp(-ya[-1] * t[-1])) * La# 计算对应A货币本金的固定利率债券价值
        Bond_B = (Rb / m + 1) * exp(-yb[0] * t[0]) * Lb# 计算对应B货币本金的浮动利率债券价值
        if trader == "A":
            swap_value = Bond_A - Bond_B * E
        else:
            swap_value = Bond_B - Bond_A / E
    return swap_value

# 一个案例
# 2020年4月1日，国内的J银行与美国的K银行达成了一笔期限为3年的固定对浮动货币互换业务，具体的合约要素如下
# （1）本金的约定：K银行在合约初始日（2020年4月1日）向J银行支付1亿美元本金，并且在合约到期日（2023年4月1日）收回该本金；
# J银行在合约初始日按照当天美元兑人民币汇率中间价7.0771向K银行支付7.0771亿元本金，同样在合约到期日收回该本金
# （2）利率的约定：交换利息的频次是每年一次。其中针对美元本金按照12个月期美元Libor利率计算利息，并且2020年4月1日该利率是1.0024%；针对人民币本金则支付固定利息
# J银行需要确定在货币互换中基于人民币本金的固定利率金额。
# 此外，在2020年6月18日和2020年7月20日这两个交易日针对不同交易方计算货币互换合约的价值。
# 根据其他信息（p329），通过Python并且分3个步骤进行计算
# 第1步：通过在9.2.4节自定义的函数swap_rate计算针对人民币本金对应的固定利率
# 2020年4月1日人民币零息利率
y_RMB_Apr1 = np.array([0.016778, 0.019062, 0.019821])

# 每年交换利息的频次
M = 1

# 合约的期限
tenor = 3

# 计算固定利率
rate_RMB = swap_rate(m = M, y = y_RMB_Apr1, T = tenor)
print("货币互换合约针对人民币本金的固定利率", round(rate_RMB, 4))

# 第2步：在Python中输入计算货币互换合约价值的相关参数
# 2020年4月1日美元兑人民币汇率
FX_Apr1 = 7.0771

# 货币互换合约的美元本金金额
par_USD = 1e8

# 货币互换合约的人民币本金金额
par_RMB = par_USD * FX_Apr1

# 2020年4月1日12个月期美元Libor
Libor_Apr1 = 0.010024

# 2020年6月18日人民币汇率
y_RMB_Jun18 = np.array([0.021156, 0.023294, 0.023811])

# 2020年6月18日美元零息利率
y_USD_Jun18 = np.array([0.0019, 0.0019, 0.0022])

# 2020年6月18日美元兑人民币汇率
FX_Jun18 = 7.0903

# 2020年7月20日人民币零息利率
y_RMB_Jul20 = np.array([0.022540, 0.024251, 0.025256])

# 2020年7月20日美元零息利率
y_USD_Jul20 = np.array([0.0014, 0.0016, 0.0018])

# 2020年7月20日美元兑人民币汇率
FX_Jul20 = 6.9928

# 货币互换合约初始日（9.2.5节已导入datetime模块并缩写为dt）
t0 = dt.datetime(2020, 4, 1)

# 货币互换合约定价日2020年6月18日
t1 = dt.datetime(2020, 6, 18)

# 货币互换合约定价日2020年7月20日
t2 = dt.datetime(2020, 7, 20)

# 2020年6月18日（定价日）距离每期利息交换日的期限数组
t1_list = np.arange(1, tenor + 1) - (t1 - t0).days / 365

# 查看结果
print(t1_list)

# 2020年7月20日（定价日）距离每期利息交换日的期限数组
t2_list = np.arange(1, tenor + 1) - (t2 - t0).days / 365

# 查看结果
print(t2_list)

# 第3步：根据第1步计算得到的人民币本金固定利率，同时运用自定义函数CCS_value以及表9-19中的信息，
# 计算2020年6月18日和2020年7月20日这两个交易日货币互换合约的价值
# 2020年6月18日J银行的货币互换合约价值
value_RMB_Jun18 = CCS_value(types = "固定对浮动货币互换", La = par_RMB, Lb = par_USD, Ra = rate_RMB, Rb = Libor_Apr1,
                            ya = y_RMB_Jun18, yb = y_USD_Jun18, E = FX_Jun18, m = M, t = t1_list, trader = "A")

# 2020年6月18日K银行的货币互换合约价值
value_USD_Jun18 = CCS_value(types = "固定对浮动货币互换", La = par_RMB, Lb = par_USD, Ra = rate_RMB, Rb = Libor_Apr1,
                            ya = y_RMB_Jun18, yb = y_USD_Jun18, E = FX_Jun18, m = M, t = t1_list, trader = "B")

# 2020年7月20日J银行的货币互换合约价值
value_RMB_Jul20 = CCS_value(types = "固定对浮动货币互换", La = par_RMB, Lb = par_USD, Ra = rate_RMB, Rb = Libor_Apr1,
                            ya = y_RMB_Jul20, yb = y_USD_Jul20, E = FX_Jul20, m = M, t = t2_list, trader = "A")

# 2020年7月20日K银行的货币互换合约价值
value_USD_Jul20 = CCS_value(types = "固定对浮动货币互换", La = par_RMB, Lb = par_USD, Ra = rate_RMB, Rb = Libor_Apr1,
                            ya = y_RMB_Jul20, yb = y_USD_Jul20, E = FX_Jul20, m = M, t = t2_list, trader = "B")

print("2020年6月18日J银行的货币互换合约价值（人民币）", round(value_RMB_Jun18, 2))
print("2020年6月18日K银行的货币互换合约价值（美元）", round(value_USD_Jun18, 2))
print("2020年7月20日J银行的货币互换合约价值（人民币）", round(value_RMB_Jul20, 2))
print("2020年7月20日K银行的货币互换合约价值（美元）", round(value_USD_Jul20, 2))
# 可看出：由于人民币的零息利率上升，加上美元兑人民币汇率上行（人民币贬值，美元升值），在2020年6月18日货币互换给J银行带来了浮亏，给K银行带来了浮盈；
# 到了2020年7月20日，由于美元兑人民币汇率出现下行（人民币升值，美元贬值），浮亏和浮盈均大幅收窄
# 故货币互换合约的价值受到两种货币的利率曲线以及即期汇率的影响



# 信用违约互换
# Python自定义函数
# 通过Python自定义一个计算信用违约互换期间现金流的函数
def CDS_cashflow(S, m, T1, T2, L, recovery, trader, event):
    """
    定义一个计算信用违约互换期间现金流的函数
    S：代表信用违约互换价差（信用保护费用）
    m：代表信用违约互换价差每年支付的频次，并且不超过2次
    T1：代表合约期限（年）
    T2：代表合约初始日距离信用事件发生日的期限长度（年），信用事件未发生则输入T2="Na"
    L：代表合约的本金
    recovery：代表信用事件发生时的回收率，信用事件未发生则输入recovery="Na"
    trader：代表交易方，输入trader=="buyer"表示买方，输入其他则表示卖方
    event：代表信用事件，输入event="N"表示合约存续期内信用事件未发生，输入其他则表示合约存续期内信用事件发生
    """
    # 为了理解代码的编写逻辑，分为以下3个步骤
    # 第1步：合约存续期内未发生信用事件时计算现金流
    if event == "N":# 当合约存续期内信用事件未发生
        n = m * T1# 计算期间现金流支付的次数
        cashflow = S * L * np.ones(n) / m# 合约期间支付信用保护费用金额的现金流
        if trader == "buyer":# 针对信用保护买方
            CF = -cashflow# 计算信用保护买方的期间现金流
        else:# 针对信用保护卖方
            CF = cashflow
    # 第2步：合约存续期内发生信用事件并且信用保护费用每年支付1次时计算现金流
    else:# 当合约存续期内信用事件发生
        default_pay = (1 - recovery) * L# 信用事件发生时合约卖方针对本金的赔偿性支付
        if m == 1:# 信用违约互换价差每年支付的频次等于1
            n = int(T2) * m + 1# 计算期间现金流支付的次数
            cashflow = S * L * np.ones(n) / m# 计算合约期间的现金流（最后一个元素后面要调整）
            spread_end = (T2 - int(T2)) * S * L# 合约最后一期（信用事件发生日）支付的信用保护费用
            cashflow[-1] = spread_end - default_pay# 合约最后一期的现金流
            if trader == "buyer":
                CF = -cashflow
            else:
                CF = cashflow
    # 第3步：合约存续期内发生信用事件并且信用保护费用每年支付2次时计算现金流
        else:# 信用违约互换价差没奶奶支付的频次等于2
            if T2 - int(T2) < 0.5:# 信用时间发生在前半年
                n = int(T2) * m + 1# 计算期间现金流支付的次数
                cashflow = S * L * np.ones(n) / m# 计算合约期间的现金流（最后一个元素后面要调整）
                spread_end = (T2 - int(T2)) * S * L# 最后一期支付的信用保护费用
                cashflow[-1] = spread_end - default_pay
                if trader == "buyer":
                    CF = -cashflow
                else:
                    CF = cashflow
            else:# 信用事件发生在后半年
                n = (int(T2) + 1) * m# 计算期间现金流支付的次数
                cashflow = S * L * np.ones(n) / m# 计算合约期间的现金流（最后一个元素后面要调整
                spread_end = (T2 - int(T2) - 0.5) * S * L# 最后一期支付的信用保护费用
                cashflow[-1] = spread_end - default_pay
                if trader == "buyer":
                    CF = -cashflow
                else:
                    CF = cashflow
    return CF# 以数组格式输出最终结果

# 结合例9-7（p331）的Python编程
# 运用自定义函数CDS_cashflow计算例9-7信用违约互换相关交易方的期间现金流，具体分为两种情形
# 情形1：合约存续期内未发生信用事件
# 信用违约互换价差（用于计算信用保护费用）
spread = 0.012

# 信用保护费用每年支付的频次
M = 1

# 信用违约互换期限（年）
tenor = 3

# 信用违约互换本金
par = 1e8

# 计算合约期间买方的现金流
cashflow_buyer1 = CDS_cashflow(S = spread, m = M, T1 = tenor, T2 = "Na", L = par, recovery = "Na", trader = "buyer",
                               event = "N")

# 计算合约期间卖方的现金流
cashflow_seller1 = CDS_cashflow(S = spread, m = M, T1 = tenor, T2 = "Na", L = par, recovery = "Na", trader = "seller",
                                event = "N")

print("未发生信用事件情形下合约期间买方的现金流", cashflow_buyer1)
print("未发生信用事件情形下合约期间卖方的现金流", cashflow_seller1)

# 情形二：合约存续期内发生信用事件，并且信用事件发生日是2020年12月1日
# 合约初始日距离信用事件发生日的期限长度（年）
T_default = 28 / 12

# 违约时的利率
rate = 0.35

# 计算合约期间买方的现金流
cashflow_buyer2 = CDS_cashflow(S = spread, m = M, T1 = tenor, T2 = T_default, L = par, recovery = rate,
                               trader = "buyer", event = "Y")

# 计算合约期间卖方的现金流
cashflow_seller2 = CDS_cashflow(S = spread, m = M, T1 = tenor, T2 = T_default, L = par, recovery = rate,
                                trader = "seller", event = "Y")

print("发生信用事件情形下合约期间买方的现金流", cashflow_buyer2)
print("发生信用事件情形下合约期间卖方的现金流", cashflow_seller2)

# 结合新案例的Python编程
# 沿用例9-7的信息，但是针对合约信息在以下两个方面做出调整：一是信用保护费用的支付频次调整为每年2次，也就是每半年1次；二是信用事件发生日变更为2021年4月1日
# 信用保护费用的支付频次调整为每年2次
M_new = 2

# 合约初始日距离新的信用事件发生日的期限长度
T_default_new = 32 / 12

# 计算合约期间买方的现金流（新的信用事件发生日）
cashflow_buyer3 = CDS_cashflow(S = spread, m = M_new, T1 = tenor, T2 = T_default_new, L = par, recovery = rate,
                               trader = "buyer", event = "Y")

# 计算合约期间卖方的现金流（新的信用事件发生日）
cashflow_seller3 = CDS_cashflow(S = spread, m = M_new, T1 = tenor, T2 = T_default_new, L = par, recovery = rate,
                                trader = "seller", event = "Y")

print("发生信用事件情况下合约期间买方的现金流（新）\n", cashflow_buyer3)
print("发生信用事件情况下合约期间卖方的现金流（新）\n", cashflow_seller3)



# 累积违约概率、边际违约概率与存活率
# 一个案例（p336）
# 运用Python计算累积违约概率、存活率、边际违约概率
# 第1步：输入相应的变量并且计算累积违约概率
# 连续复利的违约概率
h = 0.03

# 期限
T = 5

# 创建存放累积违约概率的初始数值
CDP = np.ones(T)

for t in range(1, T + 1):
    CDP[t-1] = 1 - np.exp(-h * t)# 计算累积违约概率

# 输出累积违约概率并保留小数点后4位
print(CDP.round(4))

# 第2步：计算存活率和边际违约概率
# 计算存活率
SR = 1 - CDP

# 输出存活率并保留至小数点后4位
print(SR.round(4))

# 创建存放边际违约概率的初始数组
MDP = np.ones_like(CDP)

# 第1年的边际违约概率等于同期的累积违约概率
MDP[0] = CDP[0]

for t in range(1, T):
    MDP[t] = SR[t-1] - SR[t]# 计算第2年至第5年的边际违约概率

# 输出边际违约概率并保留至小数点后4位
print(MDP.round(4))



# 信用违约互换价差
# Python自定义函数
# 通过Python自定义一个计算信用违约互换价差的函数
def CDS_spread(m, Lamda, T, R, y):
    """
    定义一个计算信用违约互换价差（年化）的函数
    m：代表信用违约互换价差（信用保护费用）每年支付的频次
    Lamda：代表连续复利的年化违约概率
    T：代表合约期限（年）
    R：代表信用事件发生时的回收率
    y：代表对应合约初始日距离每期信用保护费用支付日的期限且连续复利的零息利率，用数组格式输入
    """
    from numpy import arange, exp# 从NumPy模块导入arange函数和exp函数
    t_list = arange(m * T + 1) / m# 创建一个期限数组
    A = sum(exp(-Lamda * t_list[: -1] - y * t_list[1: ]))# 式（9-31）方括号内的分子
    B = sum(exp(-(Lamda + y) * t_list[1: ]))# 式（9-31）方括号内的分母
    spread = m * (1 - R) * (A / B - 1)# 计算信用违约互换价差
    return spread

# 一个案例
# 例9-10（p338）
# 通过自定义函数CDS_spread计算本例的信用违约互换价差
# 零息利率
zero_rate = np.array([0.021276, 0.022853, 0.024036, 0.025010, 0.025976])

# 违约回收率
recovery = 0.4

# 信用保护费用每年支付的频次
M = 1

# 合约期限（年）
tenor = 5

# 连续复利的年化违约概率
h = 0.03

# 计算信用违约互换价差
spread = CDS_spread(m = M, Lamda = h, T = tenor, R = recovery, y = zero_rate)

# 保留至小数点后4位
print("计算得到信用违约互换价差", spread.round(4))

# 敏感性分析
# 考察违约概率和违约回收率这两个重要变量发生变化时，如何影响信用违约互换价差
# 沿用例9-10信息开展名感谢分析，分为两个方面
# 一是信用违约互换价差兑违约概率的敏感性分析。当参考实体P公司的连续复利违约概率取[1%, 6%]区间的等差数列，而其余变量取值保持不变时，考察不同的违约概率对信用违约互换价差的影响
# 二是信用违约互换价差对违约回收率的敏感性分析。当参考实体P公司的违约回收率取[10%, 60%]区间的等差数列，而其余变量取值保持不变时，考察不同的违约回收率对信用违约互换价差的影响
# 第1步：当参考实体P公司的连续复利违约概率取不同数值时，计算对应的信用违约互换价差
# 违约概率数组
h_list = np.linspace(0.01, 0.06, 200)

# 创建存放信用违约互换价差（对应不同违约概率）的初始数组
spread_list1 = np.zeros_like(h_list)

for i in range(len(h_list)):
    spread_list1[i] = CDS_spread(m = M, Lamda = h_list[i], T = tenor, R = recovery, y = zero_rate)# 不同违约概率对应的信用违约互换价差

# 第2步：当参考实体P公司的违约回收率取不同数值时，计算对应的信用违约互换价差
# 违约回收率数组
r_list = np.linspace(0.1, 0.6, 200)

# 创建存放信用违约互换价差（对应不同违约回收率）的初始数组
spread_list2 = np.zeros_like(r_list)

for i in range(len(r_list)):
    spread_list2[i] = CDS_spread(m = M, Lamda = h, T = tenor, R = r_list[i], y = zero_rate)# 不同违约回收率对应的信用违约互换价差

# 第3步：将以上的计算结果进行可视化并且运用1*2子图模式，从而形象展示出违约概率、违约回收率与信用违约互换价差之间的关系
plt.figure(figsize = (11, 6))
plt.subplot(1, 2, 1)# 第1行第1列子图
plt.plot(h_list, spread_list1, "r-", lw = 2.5)
plt.xticks(fontsize = 13)
plt.xlabel(u"违约概率", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylabel(u"信用违约互换价差", fontsize = 13, rotation = 90)
plt.title(u"违约概率与信用违约互换价差的关系图", fontsize = 14)
plt.grid()
plt.subplot(1, 2, 2, sharey = plt.subplot(1, 2, 1))# 第1行第2列子图并且与第1个子图的y轴相同
plt.plot(r_list, spread_list2, "b-", lw = 2.5)
plt.xticks(fontsize = 13)
plt.xlabel(u"违约回收率", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"违约回收率与信用违约互换价差的关系图", fontsize = 14)
plt.grid()
plt.show()
# 可看出：违约概率与信用违约互换价差之间呈现正向的线性关系，违约回收率与信用互换价差之间呈现反向的线性关系
# 相比违约回收率，信用违约互换价差对违约概率的敏感程度更高
