# 第六章
# 金融机构利率
# 贷款市场报价利率（LPR）
# 对2019年8月至2020年12月期间的贷款市场报价利率数据进行可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["FangSong"]
mpl.rcParams["axes.unicode_minus"] = False
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
LPR = pd.read_excel("C:/Users/InsaneHe/desktop/Python/贷款市场报价利率(LPR)数据.xlsx", sheet_name = "Sheet1",
                    header = 0, index_col = 0)
LPR.plot(figsize = (9, 6), grid = True, fontsize = 13)
plt.ylabel(u"利率", fontsize = 11)
# 可看出无论1年期还是5年期以上的LPR，再2020年前5个月的降幅都十分明显，原因是为了有效对冲2020年初的新冠疫情对经济的负面影响



# 金融市场利率
# 银行间同业拆借利率
# 对2019年至2020年的1天期、7天期和14天期这3个常用期限的银行间同业拆借利率
# 导入外部数据
IBL = pd.read_excel("C:/Users/InsaneHe/desktop/Python/银行间同业拆借利率（2019年-2020年）.xlsx", sheet_name = "Sheet1",
                    header = 0, index_col = 0)
# 数据可视化
IBL.plot(figsize = (9, 6), grid = True, fontsize = 13)

# 增加纵坐标的标签
plt.ylabel(u"利率", fontsize = 11)
# 可看出：1天的和14天的利率波动较大，而7天的利率波动较小



# 回购利率
# 对2019年至2020年的1天期、7天期和14天期的回购定盘利率数据可视化
# 导入外部数据
FR = pd.read_excel("C:/Users/InsaneHe/desktop/Python/银行间回购定盘利率（2019年-2020年）.xlsx", sheet_name = "Sheet1",
                   header = 0, index_col = 0)

# 数据可视化
FR.plot(figsize = (9, 6), grid = True, fontsize = 13)

# 增加纵坐标的标签
plt.ylabel(u"利率", fontsize = 11)
# 可看出：1天的和14天的利率波动较大，而7天的利率波动较小



# 对2019年至2020年的1个月期、3个月期和6个月期的Shibor数据可视化
# 导入外部数据
Shibor = pd.read_excel("C:/Users/InsaneHe/desktop/Python/SHIBOR（2019年至2020年）.xlsx", sheet_name = "Sheet1",
                 header = 0, index_col = 0)

# 数据可视化
Shibor.plot(figsize = (9, 6), grid = True, fontsize = 13)

# 增加纵坐标的标签
plt.ylabel(u"利率", fontsize = 11)
# 可看出：Shibor呈现V形走势



# 人民币汇率体系
# 对2005年7月21日（7·21汇改发生日）至2020年末的美元兑人民币、欧元兑人民币、100日元兑人民币以及港元兑人民币这4个主要交易品种的日汇率中间价数据，
# 以2*2子图的方式绘制汇率走势图
# 导入外部数据
exchange = pd.read_excel("C:/Users/InsaneHe/desktop/Python/人民币汇率每日中间价（2005年7月21日至2020年年末）.xlsx", sheet_name = "Sheet1",
                         header = 0, index_col = 0)

# 可视化
exchange.plot(subplots = True, sharex = True, layout = (2, 2), figsize = (11, 9), grid = True, fontsize = 13)

# 第1张子图
plt.subplot(2, 2, 1)

# 增加第1张子图的纵坐标标签
plt.ylabel(u"汇率", fontsize = 11, position = (0, 0))
# 可看出：2005年7月21日汇改至2015年10月末，人民币对于美元是处于单边升值的通道中的，2015年10月后，人民币对美元出现有升值有贬值的双向波动；
# 由于港元采取了盯住美元的联系汇率制度，因此港元兑人民币的走势和美元兑人民币的走势相似；
# 在2005年7月21日至2020年12月期间，无论欧元还是日元，均与人民币实现了有升值也有贬值的双向波动



# 人民币汇率指数
# 对2015年11月30日（指数的起始日）至2020年末的CFETS人民币汇率指数，BIS货币篮子人民币汇率指数以及SDR货币篮子人民币汇率指数进行可视化
# 导入外部数据
index_RMB = pd.read_excel("C:/Users/InsaneHe/desktop/Python/人民币汇率指数（2005年11月30日至2020年末）.xlsx", sheet_name = "Sheet1",
                          header = 0, index_col = 0)

# 数据可视化
index_RMB.plot(figsize = (9, 6), grid = True, fontsize = 13)

# 增加纵坐标轴的标签
plt.ylabel(u"利率", fontsize = 11)
# 可看出：这3个人民币汇率指数在走势上有一定的趋同性，同时，由于不同的人民币汇率指数所参考的一篮子外汇交易币种以及币种的权重存在差异，故不同指数在走势上存在一定分化



# 利率的度量
# 利率的相对性
# 针对例6-1，可通过Python十分高效地计算除相关的结果（for循环语句）
# 本金为1万元
par = 1e4

# 2%的1年期利率
r = 0.02

# 不同的复利频次
M = [1, 2, 4, 12, 52, 365]
name = ["每年复利1次", "每半年复利1次", "每季度复利1次", "每月复利1次", "每周复利1次", "每天复利1次"]

# 建立存放1年后本息合计金额的初始数列
value = []

# 设置一个标量，用于后续的for循环语句
i = 0

# 通过for循环语句快速计算不同复利频次的本息合计金额
for m in M:
    value.append(par * (1+r/m)**m)# 将每次计算的结果存放于数列尾部
    print(name[i], "本息合计金额", round(value[i], 2))
    i = i + 1



# 将例6-1的结论进行推广，计算不同复利频次条件下本息和的函数
def FV(A, n, R, m):
    """定义一个用于计算不同复利频次本息和的函数
    A：表示初始的投资本金
    n：表示投资期限（年）
    R：表示年利率
    m：表示每年复利频次，输入m="Y"表示每年复利1次，m="S"表示每半年复利1次，m="Q"表示每季度复利1次，m="M"表示每月复利1次，m="W"表示每周复利1次，
    输入其他则表示每天复利1次
    """
    if m == "Y":# 每年复利1次
        value = A*pow(1+R, n)# 计算本息和
    elif m == "S":# 每半年复利1次
        value = A*pow(1+R/2, n*2)
    elif m == "Q":# 每季度复利1次
        value = A*pow(1+R/4, n*4)
    elif m == "M":# 每月复利1次
        value = A*pow(1+R/12, n*12)
    elif m == "W":# 每周复利1次
        value = A*pow(1+R/52, n*52)
    else:
        value = A*pow(1+R/365, n*365)
    return value

# （验证）投资期限（年）
N = 1

# 计算每周复利1次的本息和
FV_week = FV(A = par, n = N, R = r, m = "W")
print("每周复利1次得到的本息和", round(FV_week, 2))



# 复利频次与本息和的关系
# 对此关系进行可视化
# 假设初始投资本金为100元，每年复利1次的年利率是2%，投资期限是1年，考察每年复利频次从1至200所对应的1年后到期本息和
# 投资本金为100元
par_new = 100

# 生成从1到200自然数的数据
M_list = np.arange(1, 201)

# 计算到期时本息和的数组
Value_list = par_new*pow(1+r/M_list, M_list)

plt.figure(figsize = (9, 6))
plt.plot(M_list, Value_list, "r-", lw = 2.5)
plt.xlabel(u"复利频次", fontsize = 13)
plt.xlim(0, 200)
plt.ylabel(u"本息和", fontsize = 13)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"复利频次与本息和之间的关系图", fontsize = 13)
plt.grid()
plt.show()
# 复利频次增加对于本息和的边际正效应不断衰减，目测可得：当复利频次超过75后，边际正效应变得很微弱



# 利率的等价性
# 不同复利频次的利率之间存在着一种等价关系
# 比如：每半年复利1次的利率3%与每月复利1次的利率2.9814%（保留小数点后4位）可以计算得到相同金额的利息
# 等价关系1
# 设计一个函数通过复利频次m1对应的利率R1计算出等价的新复利频次m2所对应的利率R2
def R_m2(R_m1, m1, m2):
    """
    定义一个已知复利频次m1的利率，计算等价的新复利频次m2的利率的函数
    R_m1：代表对应于复利频次m1的利率
    m1：代表对应于利率R1的复利频次
    m2：代表对应于利率R2的复利频次
    """
    r = m2 * (pow(1+R_m1/m1, m1/m2)-1)# 计算对应于复利频次m2的利率
    return r

# 假定G银行对外的利率报价是3%，按半年复利，计算等价的按月复利的利率，R1=3%，m1=2，m2=12
# 按半年复利的利率
R_semiannual = 0.03

# 按半年复利的频次
m_semiannual = 2

# 按月复利的频次
m_month = 12

# 等价的按月复利的利率
R_month = R_m2(R_m1 = R_semiannual, m1 = m_semiannual, m2 = m_month)

# 保留至小数点后6位
print("计算等价的按月复利对应的利率", round(R_month, 6))



# 等价关系2
# 通过Python自定义一个已知复利频次和对应的利率，计算等价的连续复利利率的函数
def Rc(Rm, m):
    """
    定义一个已知复利频次和对应的利率，计算等价的连续复利利率的函数
    Rm：代表复利频次m的利率
    m：代表复利频次m的利率
    """
    r = m*np.log(1+Rm/m)# 计算等价的连续复利利率
    return r

# 通过Python自定义一个已知复利频次和连续复利利率，计算对应复利频次的利率的函数
def Rm(Rc, m):
    """
    定义一个已知复利频次和连续复利利率，计算对应复利频次的利率的函数
    Rc：代表连续复利利率
    m：代表复利频次
    """
    r = m*(np.exp(Rc/m)-1)# 计算复利频次m的利率
    return r

# 假定H银行对外的利率报价为4%，按季度复利，计算等价的连续复利利率，m=4，Rm=4%
# 按季度复利的利率
R1 = 0.04

# 按季度复利的频次
M1 = 4

# 计算等价的连续复利利率
R_c = Rc(Rm = R1, m = M1)
print("等价的连续复利利率", round(R_c, 6))

# 假设H银行对外的利率报价是5%，该利率是连续复利，计算等价的按月复利的利率，m=12，Rm=5%
# 连续复利的利率
R2 = 0.05

# 按月复利的频次
M2 = 12

# 计算等价的按月复利的利率
R_m = Rm(Rc = R2, m = M2)
print("等价的按月复利的利率", round(R_m, 6))



# 零息利率
# 假设期限为3年，每年复利1次的零息利率为3%
# 年利率为3%
R3 = 0.03

# 期限为3年
T = 3

# 计算3年后到期时的本息和
value_3y = FV(A = par, n = T, R = R3, m = "Y")
print("3年后到期的本息和", round(value_3y, 2))
# 金融市场上直接观察到的利率（如债券的票面利率）往往不是零息利率



# 远期利率与远期利率协议
# 远期利率的测算
# 例6-6的计算
# 第1步：输入不同期限的零息利率数据，自定义一个包含联立方程组的函数
# 本金为100元
par = 100

# 创建一个包含零息利率的数组
zero_rate = np.array([0.02, 0.022, 0.025, 0.028, 0.03])

# 创建一个包含期限的数组
T_list = np.array([1, 2, 3, 4, 5])

# 导入SciPy的子模块optimize
import scipy.optimize as sco

# 通过定义一个函数计算远期利率
def f(Rf):
    from numpy import exp# 从NumPy导入exp函数
    R2, R3, R4, R5 = Rf# 设置不同的远期利率
    # 计算第2年的远期利率的等式
    year2 = par * exp(zero_rate[0] * T_list[0]) * exp(R2 * T_list[0]) - par * exp(zero_rate[1] * T_list[1])
    # 计算第3年的远期利率的等式
    year3 = par * exp(zero_rate[1] * T_list[1]) * exp(R3 * T_list[0]) - par * exp(zero_rate[2] * T_list[2])
    # 计算第4年的远期利率的等式
    year4 = par * exp(zero_rate[2] * T_list[2]) * exp(R4 * T_list[0]) - par * exp(zero_rate[3] * T_list[3])
    # 计算第5年的远期利率的等式
    year5 = par * exp(zero_rate[3] * T_list[3]) * exp(R5 * T_list[0]) - par * exp(zero_rate[-1] * T_list[-1])
    return np.array([year2, year3, year4, year5])

# 第2步：利用SciPy子模块optimize中的fsolve函数，计算具体的数值结果
# 创建一个包含猜测的初始远期利率的数组
R0 = [0.1, 0.1, 0.1, 0.1]

# 计算远期利率
forward_rates = sco.fsolve(func = f, x0 = R0)
print("第2年的远期利率", round(forward_rates[0], 6))
print("第3年的远期利率", round(forward_rates[1], 6))
print("第4年的远期利率", round(forward_rates[2], 6))
print("第5年的远期利率", round(forward_rates[3], 6))
# 注：第1年的远期利率与1年期的零息利率是完全相同的



# 直接计算远期利率
# 自定义一个计算远期利率的函数
def Rf(R1, R2, T1, T2):
    """
    定义一个计算远期利率的函数
    R1：表示对应期限为T1的零息利率（连续复利）
    R2：表示对应期限为T2的零息利率（连续复利）
    T1：表示对应于零息利率R1的期限（年）
    T2：表示对应于零息利率R2的期限（年）
    """
    forward_rate = R2 + (R2 - R1) * T1 / (T2 - T1)# 计算远期利率
    return forward_rate

# 在以上的自定义函数Rf中输入相应的零息利率和期限等参数即可快速计算对应的远期利率
# 使用上述函数验证例6-6的结果
Rf_result = Rf(R1 = zero_rate[: 4], R2 = zero_rate[1:], T1 = T_list[: 4], T2 = T_list[1:])# 计算远期利率
print("第2年远期利率", round(Rf_result[0], 6))
print("第3年远期利率", round(Rf_result[1], 6))
print("第4年远期利率", round(Rf_result[2], 6))
print("第5年远期利率", round(Rf_result[-1], 6))



# 远期利率协议的现金流与定价
# 使用Python自定义一个计算远期利率协议现金流的函数
def Cashflow_FRA(Rk, Rm, L, T1, T2, position, when):
    """
    定义一个计算远期利率协议现金流的函数
    Rk：表示远期利率协议中约定的固定利率
    Rm：表示在T1时点观察到的[T1, T2]的参考利率
    L：表示远期利率协议的本金
    T1：表示期限
    T2：表示期限，T2大于T1
    position：表示远期协议多头或空头，输入position="long"表示多头，输入其他则表示空头
    when：表示现金流发生的具体时点，输入when="begin"表示在T1时点发生时发生现金流，输入其他则表示在T2时点发生现金流
    """
    if position == "long":# 针对远期利率协议多头
        if when == "begin":# 当现金流发生在T1时点
            cashflow = ((Rm - Rk) * (T2 - T1) * L) / (1 + (T2 - T1) * Rm)# 计算现金流
        else:
            cashflow = (Rm - Rk) * (T2 - T1) * L
    else:# 针对远期利率协议空头
        if when == "begin":
            cashflow = ((Rk - Rm) * (T2 - T1) * L) / (1 + (T2 - T1) * Rm)
        else:
            cashflow = (Rk - Rm) * (T2 - T1) * L
    return cashflow

# 计算远期利率协议现金流的案例
# 对例6-7中的远期利率协议现金流
# 远期利率协议的本金为1亿元
par_FRA = 1e8

# 远期利率协议中约定的固定利率
R_fix = 0.02

# 2020年12月31日的3个月期Shibor
Shibor_3M = 0.02756

# 设置期限1年（T1）
tenor1 = 1

# 设置期限1.25年（T2）
tenor2 = 1.25

# 远期利率协议多头（I公司）在第1.25年年末的现金流
FRA_long_end = Cashflow_FRA(Rk = R_fix, Rm = Shibor_3M, L = par_FRA, T1 = tenor1, T2 = tenor2, position = "long",
                            when = "end")

# 远期利率协议空头（J公司）在第1.25年年末的现金流
FRA_short_end = Cashflow_FRA(Rk = R_fix, Rm = Shibor_3M, L = par_FRA, T1 = tenor1, T2 = tenor2, position = "short",
                            when = "end")

# 远期利率协议多头（I公司）在第1年年末的现金流
FRA_long_begin = Cashflow_FRA(Rk = R_fix, Rm = Shibor_3M, L = par_FRA, T1 = tenor1, T2 = tenor2, position = "long",
                            when = "begin")

# 远期利率协议空头（J公司）在第1年年末的现金流
FRA_short_begin = Cashflow_FRA(Rk = R_fix, Rm = Shibor_3M, L = par_FRA, T1 = tenor1, T2 = tenor2, position = "short",
                            when = "begin")
print("I企业现金流发生在2021年3月31日的金额", round(FRA_long_end, 2))
print("J银行现金流发生在2021年3月31日的金额", round(FRA_short_end, 2))
print("I企业现金流发生在2020年12月11日的金额", round(FRA_long_begin, 2))
print("J银行现金流发生在2020年12月11日的金额", round(FRA_short_begin, 2))



# 远期利率协议定价的公式与Python自定义函数
def Value_FRA(Rk, Rf, R, L, T1, T2, position):
    """
    定义一个计算远期利率协议价值的函数
    Rk：表示远期利率协议中约定的固定利率
    Rf：表示定价日观察到的未来[T1, T2]的远期参考利率
    R：表示期限为T2的无风险利率，并且是连续复利
    L：表示远期利率协议的本金
    T1：表示期限
    T2：表示期限，T2大于T1
    position：表示远期利率协议多头或空头，输入position="long"表示多头，输入其他则表示空头
    """
    if position == "long":# 对于远期利率协议的多头
        value = L * (Rf - Rk) * (T2 - T1) * np.exp(-R * T2)# 计算远期利率协议的价值
    else:# 对于远期利率协议的空头
        value = L * (Rk - Rf) * (T2 - T1) * np.exp(-R * T2)
    return value

# 远期利率协议定价的案例
# 第1步：利用自定义函数Rf，计算2020年12月31日当天处于2021年7月1日至9月30日期间的远期的3个月期Shibor
# 6个月期Shibor
Shibor_6M = 0.02838

# 9个月期Shibor
Shibor_9M = 0.02939

# 设置期限0.5年（T1）
Tenor1 = 0.5

# 设置期限0.75年（T2）
Tenor2 = 0.75

# 计算远期的3个月期Shibor
FR_Shibor = Rf(R1 = Shibor_6M, R2 = Shibor_9M, T1 = Tenor1, T2 = Tenor2)
print("计算得到2020年12月31日远期的3个月期Shibor", round(FR_Shibor, 6))

# 第2步：利用自定义函数Value_FRA计算远期利率协议的价值
# 远期利率协议的面值
Par_FRA = 2e8

# 远期利率协议中约定的固定利率
R_fix = 0.03

# 9个月期的无风险利率
R_riskfree = 0.024477

# 计算远期利率协议空头（M公司）的协议价值
FRA_short = Value_FRA(Rk = R_fix, Rf = FR_Shibor, R = R_riskfree, L = Par_FRA, T1 = Tenor1, T2 = Tenor2, position = "short")

# 计算远期利率协议多头（N银行）的协议价值
FRA_long = Value_FRA(Rk = R_fix, Rf = FR_Shibor, R = R_riskfree, L = Par_FRA, T1 = Tenor1, T2 = Tenor2, position = "long")

print("2020年12月31日M企业的远期利率协议价值", round(FRA_short, 2))
print("2020年12月31日N银行的远期利率协议价值", round(FRA_long, 2))
# 可看出：由于远期Shibor（远期参考利率）高于固定利率，因此对协议空头M公司而言，该远期利率协议带来了浮亏；相比之下，对于协议多头N银行则该远期利率协议带来了浮盈



# 汇率报价与套利
# 汇率报价
def exchange(E, LC, FC, quote):
    """
    定义一个通过汇率计算汇兑金额的函数
    E：代表汇率报价
    LC：代表用于兑换的以本币计价的币种金额，输入LC="Na"表示未已知相关金额
    FC：代表用于兑换的以外币计价的币种金额，输入FC="Na"表示未已知相关金额
    quote：代表汇率标价方法，输入quote="direct"表示直接标价法，输入其他则表示间接标价法
    """
    if LC == "Na":# 将外币兑换为本币
        if quote == "direct":# 汇率标价方法是直接标价法
            value = FC * E# 计算兑换得到本币的金额
        else:# 汇率标价方法是间接标价法
            value = FC / E# 计算兑换得到本币的金额
    else:# 将本币兑换为外币
        if quote == "direct":
            value = LC / E# 计算兑换得到外币的金额
        else:
            value = LC * E#  计算兑换得到外币的金额
    return value

# 第2步：通过上一步自定义的函数分别计算P企业和Q企业将外币兑换成本币的金额
# 美元兑人民币的汇率
USD_RMB = 7.1277

# 英镑兑欧元的汇率
GBP_EUR = 1.1135

# P企业（中国企业）持有的美元金额
Amount_USD = 6e6

# Q企业（英国企业）持有的欧元金额
Amount_EUR = 8e6

# 兑换为人民币的金额
Amount_RMB = exchange(E = USD_RMB, LC = "Na", FC = Amount_USD, quote = "direct")

# 兑换为英镑的金额
Amount_GBP = exchange(E = GBP_EUR, LC = "Na", FC = Amount_EUR, quote = "indirect")

print("P企业将600万美元兑换成人民币的金额（单位：元）", round(Amount_RMB, 2))
print("Q企业将800万欧元兑换成英镑的金额（单位：英镑）", round(Amount_GBP, 2))



# 三角套利（P192）
# 自定义一个计算三角套利收益并显示套利路径的函数
def tri_arbitrage(E1, E2, E3, M, A, B, C):
    """
    定义一个计算三角套利收益并显示套利路径的函数：
    E1：代表A货币兑B货币的汇率，以若干个A货币表示1个单位的B货币
    E2：代表B货币兑C货币的汇率，以若干个B货币表示1个单位的C货币
    E3：代表A货币兑C货币的汇率，以若干个A货币表示1个单位的C货币
    M：代表以A货币计价的初始本金
    A：代表A货币的名称，例如输入A = "人民币"就表示A货币是人民币
    B：代表B货币的名称，例如输入B = "美元"就表示B货币是美元
    C：代表C货币的名称，例如输入C = "欧元"就表示C货币是欧元
    """
    E3_new = E1 * E2# 计算A货币兑C货币的交叉汇率
    if E3_new > E3:# 当交叉汇率高于直接的汇率报价
        profit = M * (E3_new / E3 - 1)# 套利收益
        sequence = ["三角套利的路径：", A, "→", C, "→", B, "→", A]# 设定套利的路径
    elif E3_new < E3:# 当交叉汇率低于直接的汇率报价
        profit = M * (E3 / E3_new - 1)
        sequence = ["三角套利的路径：", A, "→", B, "→", C, "→", A]
    else:# 当交叉汇率等于直接的汇率报价
        profit = 0
        sequence = ["三角套利的路径：不存在"]
    return [profit, sequence]# 输出包含套利收益和套利路径的列表
# 注：3个汇率参数E1，E2，E3有其特定的标价输入规则，因此当拟输入的汇率的标价规则与自定义函数中需要输入的标价规则不一致时，就要对拟输入的汇率做相应的调整（取倒数）后再输入

# 验证
# 美元兑人民币的汇率
USD_RMB = 7.0965

# 美元兑卢布的汇率
USD_RUB = 68.4562

# 人民币兑卢布的汇率
RMB_RUB = 9.7150

# R公司拥有的以人民币计价的初始本金
value_RMB = 1e8

# 测算套利收益和套利路径
arbitrage = tri_arbitrage(E1 = USD_RMB, E2 = 1 / USD_RUB, E3 = 1 / RMB_RUB, M = value_RMB, A = "人民币", B = "美元", C = "卢布")
print("三角套利的收益", round(arbitrage[0], 2))
print(arbitrage[1])



# 远期汇率与远期外汇合约
# 远期汇率的测算
# 自定义一个计算远期汇率的函数（P194）
def FX_forward(spot, r_A, r_B, T):
    """
    定义一个计算远期汇率的函数，并且两种货币分别是A货币和B货币
    spot：代表即期汇率，标价方式是以若干个单位的A货币表示1个单位的B货币
    r_A：代表A货币的无风险利率，并且每年复利1次
    r_B：代表B货币的无风险利率，并且每年复利1次
    T：代表远期汇率的期限，并且以年为单位
    """
    forward = spot * (1 + r_A * T) / (1 + r_B * T)# 计算远期汇率
    return forward

# 验证
# 假定U银行要在2020年6月5日计算期限分别为1个月，3个月，6个月和1年的美元兑人民币远期汇率，当天的即期汇率为7.0965。
# 同时，用Shibor代表人民币的无风险利率，用美元Libor代表美元的无风险利率（P195）
# 即期利率
FX_spot = 7.0965

# 4个不同期限的数据
Tenor = np.array([1/12, 3/12, 6/12, 1.0])

# Shibor
Shibor = np.array([0.015820, 0.015940, 0.016680, 0.019030])

# Libor利率
Libor = np.array([0.001801, 0.003129, 0.004813, 0.006340])

# 与期限数组形状相同的初始远期汇率数组
FX_forward_list = np.zeros_like(Tenor)

# 运用for语句快速计算不同期限的远期利率
for i in range(len(Tenor)):
    FX_forward_list[i] = FX_forward(spot = FX_spot, r_A = Shibor[i], r_B = Libor[i], T = Tenor[i])# 计算不同期限的远期汇率

print("1个月期的美元兑人民币远期汇率", round(FX_forward_list[0], 4))
print("3个月期的美元兑人民币远期汇率", round(FX_forward_list[1], 4))
print("6个月期的美元兑人民币远期汇率", round(FX_forward_list[2], 4))
print("1年期的美元兑人民币远期汇率", round(FX_forward_list[-1], 4))



# 抵补套利
# 自定义一个计算抵补套利收益并显示套利路径的函数
def cov_arbitrage(S, F, M_A, M_B, r_A, r_B, T, A, B):
    """
    定义一个计算抵补套利收益并显示套利路径的函数，并且两种货币分别是A货币和B货币
    spot：代表即期汇率，以若干个单位的A货币表示1个单位的B货币
    forward：代表外汇市场报价的远期汇率，标价方式与即期汇率的一致
    M_A：代表借入A货币的本金，输入M_A = "Na"表示未已知相关金额
    M_B：代表借入B货币的本金，输入M_B = "Na"表示未已知相关金额
    r_A：代表A货币的无风险利率，并且每年复利1次
    r_B：代表B货币的无风险利率，并且每年复利1次
    T：代表远期汇率的期限，并且以年为单位
    A：代表A货币的名称，例如输入A = "人民币"表示A货币是人民币
    B：代表B货币的名称，例如输入B = "美元"表示B货币是美元
    """
# 第1步：计算均衡远期汇率并且当均衡远期汇率小于实际汇率时
    F_new = S * (1 + r_A * T) / (1 + r_B * T)# 计算均衡远期汇率
    if F_new < F:# 均衡远期汇率小于实际汇率
        if M_B == "Na":# 借入A货币的本金
            profit = M_A * (1 + T * r_B) * F / S - M_A * (1 + T * r_A)# 计算初始借入A货币抵补套利的套利收益
            if profit > 0:# 套利收益大于0
                sequence = ["套利路径如下",
                            "（1）初始时刻借入的货币名称：", A,
                            "（2）按照即期汇率兑换后并投资的货币名称：", B,
                            "（3）按照远期汇率在投资结束时兑换后的货币名称：", A,
                            "（4）偿还初始时刻的借入资金"]
            else:# 套利收益小于0
                sequence = ["不存在套利机会"]
        else:# 借入B货币的本金
            profit = M_B * S * (1 + T * r_A) / F - M_B * (1 + T * r_B)# 计算初始借入B货币抵补套利的套利收益
            if profit > 0:# 套利收益大于0
                sequence = ["套利路径如下",
                            "（1）初始时刻借入的货币名称：", B,
                            "（2）按照即期汇率兑换后并投资的货币名称：", A,
                            "（3）按照远期汇率在投资结束时兑换后的货币名称：", B,
                            "（4）偿还初始时刻的借入资金"]
            else:
                sequence = ["不存在套利机会"]
# 第2步：当均衡远期汇率大于实际远期汇率时
    elif F_new > F:# 均衡远期汇率大于实际远期汇率
        if M_B == "Na":# 借入A货币的本金
            profit = M_A * (1 + T * r_B) * F / S - M_A * (1 + T * r_A)# 计算初始借入A货币抵补套利的套利收益
            if profit > 0:# 套利收益大于0
                sequence = ["套利路径如下",
                            "（1）初始时刻借入的货币名称：", A,
                            "（2）按照即期汇率兑换后并投资的货币名称：", B,
                            "（3）按照远期汇率在投资结束时兑换后的货币名称：", A,
                            "（4）偿还初始时刻的借入资金"]
            else:# 套利收益小于0
                sequence = ["不存在套利机会"]
        else:# 借入B货币的本金
            profit = M_B * S * (1 + T * r_A) / F - M_B * (1 + T * r_B)# 计算初始借入B货币抵补套利的套利收益
            if profit > 0:# 套利收益大于0
                sequence = ["套利路径如下",
                            "（1）初始时刻借入的货币名称：", B,
                            "（2）按照即期汇率兑换后并投资的货币名称：", A,
                            "（3）按照远期汇率在投资结束时兑换后的货币名称：", B,
                            "（4）偿还初始时刻的借入资金"]
            else:# 套利收益小于0
                sequence = ["不存在套利机会"]
# 第3步：当均衡远期汇率等于实际远期汇率时
    else:# 均衡远期汇率等于实际远期汇率
        if M_B == "Na":# 借入A货币的本金
            profit = 0
            sequence = ["不存在套利机会"]
        else:# 借入B货币的本金
            profit = 0
            sequence = ["不存在套利机会"]
    return [profit, sequence]

# 验证
# 拆入人民币金额
value_RMB = 1e8

# 拆入美元金额
value_USD = 1.4e7

# 3个月期Shibor
Shibor_3M = 0.01594

# 3个月期Libor
Libor_3M = 0.003129

# 3个月的期限（以年为单位）
tenor = 3/12

# 2020年6月5日美元兑人民币的即期汇率
FX_spot = 7.0965

# 2020年6月5日报价的美元兑人民币远期汇率
FX_forward = 7.1094

# 初始时刻通过借入人民币开展抵补套利
arbitrage_RMB = cov_arbitrage(S = FX_spot, F = FX_forward, M_A = value_RMB, M_B = "Na", r_A = Shibor_3M, r_B = Libor_3M,
                              T = tenor, A = "人民币", B = "美元")

print("借入人民币1亿元开展抵补套利的收益（元）", round(arbitrage_RMB[0], 2))
print(arbitrage_RMB[1])# 输出套利路径

# 初始时刻通过借入美元开展抵补套利
arbitrage_USD = cov_arbitrage(S = FX_spot, F = FX_forward, M_A = "Na", M_B = value_USD, r_A = Shibor_3M, r_B = Libor_3M,
                              T = tenor, A = "人民币", B = "美元")

print("借入1400万美元开展抵补套利的收益（美元）", round(arbitrage_USD[0], 2))
print(arbitrage_USD[1])# 输出套利路径



# 远期外汇合约的定价
# 自定义一个计算远期外汇合约价值的函数（P202）
def Value_FX_Forward(F1, F2, S, par, R, t, pc, vc, position):
    """
    定义一个计算远期外汇合约价值的函数，并且两种货币分别是A货币和B货币
    F1：代表合约初始日约定的远期汇率，以若干个单位的A货币表示1个单位的B货币
    F2：代表合约初始日的远期汇率，标价方式与F1相同
    S：代表合约定价日的即期汇率，标价方式与F1相同
    par：代表合约本金，并且本金的计价货币需要与确定币种参数pc保持一致
    R：代表非合约本金计价货币的无风险利率（连续复利），比如本金是A货币，则该利率就是B货币的无风险利率
    t：代表合约的剩余期限，并且以年为单位
    pc：代表合约本金的币种，输入pc = "A"表示选择A货币，输入其他则表示选择B货币
    vc：代表合约价值的币种，输入vc = "A"表示选择A货币，输入其他则表示选择B货币
    position：代表合约的头寸方向，输入position = "long"表示合约多头，输入其他则表示合约空头
    """
    from numpy import exp# 从NumPy导入exp函数
    if pc == "A":# 合约本金以A货币计价
        if position == "long":# 针对合约多头
            if vc == "A":# 合约价值用A货币计价
                value = S * (par / F2 - par / F1) * exp(-R * t)# 计算合约价值
            else:# 合约价值用B货币计价
                value = (par / F2 -par / F1) * exp(-R * t)
        else:# 针对合约空头
            if vc == "A":
                value = S * (par / F1 - par / F2) * exp(-R * t)
            else:
                value = (par / F1 - par / F2) * exp(-R * t)
    else:# 合约本金以B货币计价
        if position == "long":
            if vc == "A":
                value = (par * F2 - par * F1) * exp(-R * t)
            else:
                value = (par * F2 - par * F1) * exp(-R * t) / S
        else:
            if vc == "A":
                value = (par * F1 - par * F2) * exp(-R * t)
            else:
                value = (par * F1 - par * F2) * exp(-R * t) / S
    return value

# 验证（P202）
# 第1步：计算合约初始日（2020年2月28日）以及合约定价日（2020年5月28日）的远期汇率
# 远期外汇合约的本金（人民币）
par_RMB = 1e8

# 2020年2月28日的即期汇率
FX_spot_Feb28 = 7.0066

# 2020年2月28日的6个月期Shibor利率
Shibor_6M_Feb28 = 0.0256

# 2020年2月28日的6个月期Libor利率
Libor_6M_Feb28 = 0.013973

# 远期外汇合约的整个期限（年）
T1 = 6/12

# 计算2020年2月28日的6个月期远期汇率
FX_forward_Feb28 = FX_forward(spot = FX_spot_Feb28, r_A = Shibor_6M_Feb28, r_B = Libor_6M_Feb28, T = T1)

print("2020年2月28日的6个月期美元兑人民币远期汇率", FX_forward_Feb28)

# 2020年5月28日的即期汇率
FX_spot_May28 = 7.1277

# 2020年5月28日的3个月期Shibor
Shibor_3M_May28 = 0.0143

# 2020年5月28日的3个月期Libor
Libor_3M_May28 = 0.0035

# 2020年5月28日连续复利的美元无风险利率
rate_USD = 0.003494

# 远期外汇合约的剩余期限（年）
T2 = 3/12

# 计算2020年5月28日的3个月期远期汇率
FX_forward_May28 = FX_forward(spot = FX_spot_May28, r_A = Shibor_3M_May28, r_B = Libor_3M_May28, T = T2)

print("2020年5月28日的3个月期美元兑人民币远期汇率", FX_forward_May28)

# 第2步：计算合约定价日远期外汇合约的价格
# 合约空头的价值
value_short = Value_FX_Forward(F1 = FX_forward_Feb28, F2 = FX_forward_May28, S = FX_spot_May28, par = par_RMB,
                               R = rate_USD, t = T2, pc = "A", vc = "A", position = "short")

# 合约多头的价值
value_long = Value_FX_Forward(F1 = FX_forward_Feb28, F2 = FX_forward_May28, S = FX_spot_May28, par = par_RMB,
                               R = rate_USD, t = T2, pc = "A", vc = "B", position = "long")

print("合约空头（V公司）在2020年5月28日的远期外汇合约价值（元）", round(value_short, 2))
print("合约多头（W银行）在2020年5月28日的远期外汇合约价值（美元）", round(value_long, 2))
