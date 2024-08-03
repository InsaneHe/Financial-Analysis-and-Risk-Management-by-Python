# 第七章
# 债券市场概览
# 通过导入外部数据并且运用Python绘制2010年至2020年债券存量与国内生产总值（GDP）对比的走势图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["FangSong"]
mpl.rcParams["axes.unicode_minus"] = False
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# 导入外部数据
bond_GDP = pd.read_excel()

# 可视化
bond_GDP.plot(kind = "bar", figsize = (9, 6), fontsize = 13, grid = True)

# 增加纵坐标标签
plt.ylabel(u"金额", fontsize = 11)
# 可看出：债券存量规模不断追赶GDP规模，并最终在2020年成功超越GDP



# 债券交易场所
# 导入外部数据，用Python绘制出2020年年末存量债券在不同交易市场分布情况的饼图
# 导入外部数据
bond = pd.read_excel("C:/Users/InsaneHe/desktop/Python/2020年年末存量债券的市场分布情况.xlsx", sheet_name = "Sheet1",
                     header = 0, index_col = 0)

plt.figure(figsize = (9, 6))
plt.pie(x = bond["债券余额（亿元）"], labels = bond.index)
plt.axis("equal")# 使饼图是一个圆形
plt.legend(loc = 2, fontsize = 13)# 图例在左上方
plt.title(u"2020年末存量债券的市场分布图", fontsize = 13)
plt.show()



# 债券定价与债券收益率
# 自定义一个基于单一贴现率计算债券价格的函数
def Bondprice_onediscount(C, M, m, y, t):
    """
    定义一个基于单一贴现率计算债券价格的函数
    C：代表债券的票面利率，如果输入0则表示零息债券
    M：代表债券的本金（面值）
    m：代表债券票息每年支付的频次
    y：代表单一贴现率
    t：代表定价日至后续每一期票息支付日的期限长度，用数组格式输入；零息债券可直接输入数字
    """
    if C == 0:# 针对零息债券
        price = np.exp(-y * t) * M# 计算零息债券的价格
    else:# 针对带息票债券
        coupon = np.ones_like(t) * M * C / m# 创建每一期票息金额的数组
        NPV_coupon = np.sum(coupon * np.exp(-y * t))# 计算每一期票息在定假日的现值之和
        NPV_par = M * np.exp(-y * t[-1])# 计算本金在定价日的现值
        price = NPV_coupon + NPV_par# 计算定价日的债券价格
    return price

# 案例
# 例1：2020年6月5日财政部发行了”20贴现国债27“（债券代码209927），面值为100元，期限为0.5年，到期一次支付本金（零息债券），
# 起息日为2020年6月8日，到期日2020年12月7日，假设定价日为2020年6月8日，贴现率1.954%且连续复利
# 20贴现国债27的票面利率
C_TB2027 = 0

# 债权本金（后续案例涉及的本金将直接调用该变量）
par = 100

# 20贴现国债27的期限
T_TB2027 = 0.5

# 20贴现国债27每年支付票息的频次
m_TB2027 = 0

# 20贴现国债27的贴现利率
y_TB2027 = 0.01954

# 计算债券价格
value_TB2027 = Bondprice_onediscount(C = C_TB2027, M = par, m = m_TB2027, y = y_TB2027, t = T_TB2027)

print("2020年6月8日20贴现国债27的价格", round(value_TB2027, 4))

# 例2：2020年5月20日财政部发行”20附息国债06“（债券代码200006），面值100元，期限10年，票面利率2.68%（带票息债券），每年付息2次，
# 起息日2020年5月21日，到期日2030年5月20日，假设定价日2020年5月21日，贴现率2.634%（连续复利），债券存续期内有共计20期的票息支付（P213）
# 20附息国债06的票面利率
C_TB2006 = 0.0268

# 20附息国债06每年支付票息的频次
m_TB2006 = 2

# 20附息国债06的贴现利率
y_TB2006 = 0.02634

# 20附息国债06的期限
T_TB2006 = 10

# 定价日至每期票息支付日的期限数组
Tlist_TB2006 = np.arange(1, m_TB2006 * T_TB2006 + 1) / m_TB2006

# 查看输出的结果
print(Tlist_TB2006)

# 计算债券价格
value_TB2006 = Bondprice_onediscount(C = C_TB2006, M = par, m = m_TB2006, y = y_TB2006, t = Tlist_TB2006)

print("2020年5月21日20附息国债06的价格", round(value_TB2006, 4))



# 债券到期收益率
# 计算带票息债券的到期收益率
# 例：2009年6月10日财政部发行”09附息国债11“（债券代码为090011），期限15年，面值100元，票面利率3.69%，每半年付息1次，
# 2020年6月11日该债券市场价格为104.802元，剩余期限4年，剩余共8次票息支付
# 第1步：通过Python自定义一个计算债券到期收益率（连续复利）的函数
def YTM(P, C, M, m, t):
    """
    P：代表观察到的债券市场价格
    C：代表债券的票面利率
    M：代表债券的本金
    m：代表债券票息每年支付的频次
    t：代表定价日至后续每一期票息支付日的期限长度，用数组格式输入；零息债券可直接输入数字
    """
    import scipy.optimize as so# 导入SciPy的子模块optimize
    def f(y):# 需要再自定义一个函数
        coupon = np.ones_like(t) * M * C / m# 创建每一期票息金额的数组
        NPV_coupon = np.sum(coupon * np.exp(-y * t))# 计算每一期票息在定价日的现值之和
        NPV_par = M * np.exp(-y * t[-1])# 计算本金在定价日的现值
        value = NPV_coupon + NPV_par# 定价日的债券现金流现值之和
        return value - P# 债券现金流现值之和减去债券市场价格
    if C == 0:# 针对零息债券
        y = (np.log(M / P)) / t# 计算零息债券的到期收益率
    else:# 针对带票息债券
        y = so.fsolve(func = f, x0 = 0.1)# 第2个参数使任意输入的初始值
    return y

# 第2步：运用第1步中自定义的计算债券到期收益率（连续复利）的函数YTM，求解出上例的到期收益率
# 09附息国债11的市场价格
P_TB0911 = 104.802

# 09附息国债11的票面利率
C_TB0911 = 0.0369

# 09附息国债11票息支付的频次
m_TB0911 = 2

# 09附息国债11的剩余期限
T_TB0911 = 4

# 定价日至每期票息支付日的期限数组
Tlist_TB0911 = np.arange(1, m_TB0911 * T_TB0911 + 1) / m_TB0911

# 计算到期收益率（数组格式）
Bond_yield = YTM(P = P_TB0911, C = C_TB0911, M = par, m = m_TB0911, t = Tlist_TB0911)

# 转换为单一的浮点型
Bond_yield = float(Bond_yield)

print("2020年6月11日09附息国债11的到期收益率", round(Bond_yield, 6))

# 第3步：对计算结果进行验证，即使用第2步计算得到的债券到期收益率并结合自定义函数Bondprice_onediscount计算09附息国债11的债券价格
# 计算债券价格
price = Bondprice_onediscount(C = C_TB0911, M = par, m = m_TB0911, y = Bond_yield, t = Tlist_TB0911)
print("09附息国债11的债券价格（用于验证）", round(price, 4))
# 说明第2步计算结果正确



# 基于不同期限贴现率的债券定价
# 通过Python自定义一个基于不同期限贴现利率计算债券价格的函数
def Bondprice_diffdiscount(C, M, m, y, t):
    """
    定义一个基于不同期限贴现利率计算债券价格的函数
    C：代表债券的票面利率，如果输入0则表示零息债券
    M：代表债券的本金
    m：代表债券票息每年支付的频次
    y：代表不同期限的贴现利率，用数组格式输入；零息债券可直接输入数字
    t：代表定价日至后续每一期票息支付日的期限长度，用数组格式输入；零息债券可直接输入数字
    """
    if C == 0:# 针对零息债券
        price = np.exp(-y * t) * M# 计算零息债券的价格
    else:# 针对带票息债券
        coupon = np.ones_like(y) * M * C / m# 创建每一期票息金额的数组
        NPV_coupon = np.sum(coupon * np.exp(-y * t))# 计算每一期票息在定价日的现值之和
        NPV_par = M * np.exp(-y[-1] * t[-1])# 计算本金在定价日的现值
        price = NPV_coupon + NPV_par# 计算定价日的债券价格
    return price



# 通过票息剥离法计算零息利率
# 通过Python计算例7-4中5只债券的零息利率（P216）
# 第1步：输入相关已知的参数并通过Python自定义一个包含联立方程组的函数
# 不同期限债券价格
P = np.array([99.5508, 99.0276, 100.8104, 102.1440, 102.2541])

# 债券的期限结构
T = np.array([0.25, 0.5, 1.0, 1.5, 2.0])

# 债券票面利率数组
C = np.array([0, 0, 0.0258, 0.0357, 0.0336])

# 第4只和第5只债券的付息频次
m = 2

# 通过自定义函数求解零息利率
def f(R):
    from numpy import exp# 从NumPy中导入exp函数
    R1, R2, R3, R4, R5 = R# 不同期限的零息利率
    B1 = P[0] * exp(R1 * T[0]) - par# 用第1只债券计算零息利率的公式
    B2 = P[1] * exp(R2 * T[1]) - par# 用第2只债券计算零息利率的公式
    B3 = P[2] * exp(R3 * T[2]) - par * (1 + C[2])# 用第3只债券计算零息利率的公式
    B4 = par * (C[3] * exp(-R2 * T[1]) / m + C[3] * exp(-R3 * T[2]) / m +
                (1+C[3] / m) * exp(-R4 * T[3])) - P[3]# 用第4只债券计算零息利率的公式
    B5 = par * (C[-1] * exp(-R2 * T[1]) / m + C[-1] * exp(-R3 * T[2]) / m + C[-1] * exp(-C[3] * T[3]) / m +
                (1 + C[-1] / m) * exp(-R5 * T[-1])) - P[-1]# 用第5只债券计算零息利率的公式
    return np.array([B1, B2, B3, B4, B5])

# 第2步：运用SciPy的子模版optimize中的函数fsolve求解第1步中的联立方程组
# 导入SciPy的子模块optimize
import scipy.optimize as so

# 初始猜测的零息利率
r0 = [0.1, 0.1, 0.1, 0.1, 0.1]

# 计算不同期限的零息利率
rates = so.fsolve(func = f, x0 = r0)

print("0.25年期的零息利率（连续复利）", round(rates[0], 6))
print("0.5年期的零息利率（连续复利）", round(rates[1], 6))
print("1年期的零息利率（连续复利）", round(rates[2], 6))
print("1.5年期的零息利率（连续复利）", round(rates[3], 6))
print("2年期的零息利率（连续复利）", round(rates[4], 6))

# 第3步：对计算得到的零息利率进行可视化
plt.figure(figsize = (9, 6))
plt.plot(T, rates, "b-")
plt.plot(T, rates, "ro")
plt.xlabel(u"期限（年）", fontsize = 13)
plt.xticks(fontsize = 13)
plt.ylabel(u"利率", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"运用票息剥离法得到的零息曲线", fontsize = 13)
plt.grid()
plt.show()



# 插值处理
# 从上图看出对应于期限0.75年，1.25年和1.75年的零息利率是缺失的，当债券市场缺少恰好等于这些期限的债券时通常替代做法为基于已有的零息利率数据进行插值处理
# SciPy子模块interpolate中的interp1d函数
# 沿用上例得到的零息利率，使用3阶样条曲线插值法计算期限为0.75年，1.25年和1，75年的零息利率并绘制相应的零息曲线
# 第1步：选择插值的具体方法并进行相应的计算
# 导入SciPy的子模块interpolate
import scipy.interpolate as si

# 运用已有数据构建插值函数并且运用3阶样条曲线插值法
func = si.interp1d(x = T, y = rates, kind = "cubic")

# 创建包含0.75年，1.25年和1.75年期限的数组
T_new = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

# 计算基于插值法的零息利率
rates_new = func(T_new)

# 运用for语句快速输出相关结果
for i in range(len(T_new)):
    print(T_new[i], "年期限的零息利率", round(rates_new[i], 6))

# 第2步：对计算得到的结果进行可视化
plt.figure(figsize = (9, 6))
plt.plot(T_new, rates_new, "o")
plt.plot(T_new, rates_new, "-")
plt.xlabel(u"期限（年）", fontsize = 13)
plt.xticks(fontsize = 13)
plt.ylabel(u"利率", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"基于3阶样条曲线插值方法得到的零息曲线", fontsize = 13)
plt.grid()
plt.show()



# 运用零息利率对债券定价
# 假定在2020年6月8日，国内债券市场有一只债券，本金100元，票面利率3.6%，每年支付票息4次，每季度支付票息100*3.6%/4=0.9元，剩余期限2年，用上例的零息利率对其定价
# 债券的票面利率
C_new = 0.036

# 债券票息支付频次
m_new = 4

# 计算债券价格
price_new = Bondprice_diffdiscount(C = C_new, M = par, m = m_new, y = rates_new, t = T_new)
print("基于不同期限的贴现利率计算债券价格", round(price_new, 4))



# 衡量债券利率风险的线性指标——久期
# 麦考利久期
# 票面利率下降时，债券的麦考利久期上升
# 用Python自定义一个计算债券麦考利久期的函数
def Mac_duration(C, M, m, y, t):
    """
    定义一个计算债券麦考利久期的函数
    C：代表债券的票面利率
    M：代表债券的面值
    m：代表债券票息每年支付的频次
    y：代表债券的到期收益率（连续复利）
    t：代表定价日至后续每一期现金流支付日的期限长度，用数组格式输入；零息债券可直接输入数字
    """
    if C == 0:# 针对零息债券
        duration = t# 计算零息债券的麦考利久期
    else:# 针对带息票债券
        coupon = np.ones_like(t) * M * C / m# 创建每一期票息金额的数组
        NPV_coupon = np.sum(coupon * np.exp(-y * t))# 计算每一期票息在定价日的现值之和
        NPV_par = M * np.exp(-y * t[-1])# 计算本金在定价日的现值
        Bond_value = NPV_coupon + NPV_par# 计算定价日的债券价格
        cashflow = coupon# 现金流数组并初始设定等于票息
        cashflow[-1] = M * (1 + C/m)# 现金流数组最后的元素调整为票息与本金之和
        weight = cashflow * np.exp(-y * t) / Bond_value# 计算时间的权重
        duration = np.sum(t * weight)# 计算带票息债券的麦考利久期
    return duration

# 一个案例
# 2020年6月12日，“09附息国债11”的剩余期限为4年，到期日为2024年6月11日，面值为100元，票面利率为3.69%，票息为每年支付2次（半年1次），到期收益率为2.4%（连续复利）
# 09附息国债11的票面利率
C_TB0911 = 0.0369

# 09附息国债11的票息支付频次
m_TB0911 = 2

# 09附息国债11的到期收益率
y_TB0911 = 0.024

# 09附息国债11的剩余期限（年）
T_TB0911 = 4

# 09附息国债11现金流支付期限数组
Tlist_TB0911 = np.arange(1, m_TB0911 * T_TB0911 + 1) / m_TB0911

# 查看输出结果
print(Tlist_TB0911)

# 计算麦考利久期
D1_TB0911 = Mac_duration(C = C_TB0911, M = par, m = m_TB0911, y = y_TB0911, t = Tlist_TB0911)
print("2020年6月12日09附息国债11的麦考利久期", round(D1_TB0911, 4))



# 麦考利久期的其他重要公式（P223）
# 沿用上例的09附息国债11的债券信息，假定2020年6月12日的债券到期收益率（连续复利）从2.4%增加至2.45%，分别用麦考利久期和债券定价公式计算债券价格的变化金额
# 方法一：麦考利久期
# 计算到期收益率变化前的债券价格
price_before = Bondprice_onediscount(C = C_TB0911, M = par, m = m_TB0911, y = y_TB0911, t = Tlist_TB0911)
print("2020年6月12日到期收益率变化前的09附息国债11价格", round(price_before, 4))

# 债券到期收益率变化金额
y_change = 0.0005

# 用麦考利久期计算债券价格变化金额
price_change1 = -D1_TB0911 * price_before * y_change
print("用麦考利久期计算09附息国债11的价格变化金额", round(price_change1, 4))

# 用麦考利久期近似计算到期收益率变化后的债券价格
price_new1 = price_before + price_change1
print("用麦考利久期近似计算到期收益率变化后的09附息国债11价格", round(price_new1, 4))

# 方法二：运用债券定价公式计算精确的债券价格
# 精确计算到期收益率变化后的债券价格
price_new2 = Bondprice_onediscount(C = C_TB0911, M = par, m = m_TB0911, y = y_TB0911 + y_change, t = Tlist_TB0911)
print("精确计算2020年6月12日到期收益率变化后的09附息国债11价格", round(price_new2, 4))
# 两个方法计算出的结果较为接近



# 票面利率、到期收益率与麦考利久期之间的关系
# 沿用上例中“09附息国债11”债券信息并分为以下2种情形考察
# 情形1：当09附息国债11的票面利率是在[2%, 6%]区间进行等差取值，同时保持其他参数不变时，计算不同票面利率对应的麦考利久期，并且将票面利率与麦考利久期的关系可视化
# 情形2：当09附息国债11的到期收益率是在[1%, 5%]区间进行等差取值，同时保持其他参数不变时，计算不同到期收益率对应的麦考利久期，并且将到期收益率与麦考利久期的关系可视化
# 第1步：依次计算不同票面利率、不同到期收益率对应的麦考利久期
# 票面利率在[2%, 6%]区间进行等差取值
C_list = np.linspace(0.02, 0.06, 200)

# 到期收益率在[1%, 5%]区间进行等差取值
y_list = np.linspace(0.01, 0.05, 200)

# 创建存放对应不同票面利率的麦考利久期初始数据
D_list1 = np.ones_like(C_list)

# 创建存放对应不同到期收益率的麦考利久期初始数据
D_list2 = np.ones_like(y_list)

# 用for语句计算对应不同票面利率的麦考利久期
for i in range(len(C_list)):
    D_list1[i] = Mac_duration(C = C_list[i], M = par, m = m_TB0911, y = y_TB0911, t = Tlist_TB0911)

# 用for语句计算对应不同到期收益率的麦考利久期
for i in range(len(y_list)):
    D_list2[i] = Mac_duration(C = C_TB0911, M = par, m = m_TB0911, y = y_list[i], t = Tlist_TB0911)

# 第2步：将票面利率、到期收益率与麦考利久期的关系进行可视化并且用1*2子图模式展示
plt.figure(figsize = (11, 6))
plt.subplot(1, 2, 1)# 第1行第1列的子图
plt.plot(C_list, D_list1, "r-", lw = 2.5)
plt.xticks(fontsize = 13)
plt.xlabel(u"票面利率", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylabel(u"麦考利久期", fontsize = 13)
plt.title(u"票面利率与麦考利久期的关系图", fontsize = 14)
plt.grid()
plt.subplot(1, 2, 2, sharey = plt.subplot(1, 2, 1))# 与第1个子图的y轴同刻度
plt.plot(y_list, D_list2, "b-", lw = 2.5)
plt.xticks(fontsize = 13)
plt.xlabel(u"到期收益率", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"到期收益率与麦考利久期的关系图", fontsize = 14)
plt.grid()
plt.show()
# 可看出：无论票面利率还是到期收益率，均与麦考利久期呈现反向关系；此外，相比到期收益率，麦考利久期对票面利率显得更加敏感



# 修正久期
# 运用Python自定义一个计算债券修正久期的函数
def Mod_Duration(C, M, m1, m2, y, t):
    """
    定义一个计算债券修正久期的函数
    C：代表债券的票面利率
    M：代表债券的面值
    m1：代表债券票息每年支付的频次
    m2：代表债券到期收益率每年复利频次，通常m2等于m1
    y：代表每年复利m2次的到期收益率
    t：代表定价日至后续每一期现金流支付日的期限长度，用数组格式输入；零息债券可直接输入数字
    """
    if C == 0:# 针对零息债券
        Macaulay_duration = t# 计算零息债券的麦考利久期
    else:# 针对带息票债券
        r = m2 * np.log(1 + y / m2)# 计算等价的连续复利到期收益率
        coupon = np.ones_like(t) * M * C / m1# 创建每一期票息金额的数组
        NPV_coupon = np.sum(coupon * np.exp(-r * t))# 计算每一期票息在定价日的现值之和
        NPV_par = M * np.exp(-r * t[-1])# 计算本金在定价日的现值
        price = NPV_coupon + NPV_par# 计算定价日的债券价格
        cashflow = coupon# 现金流数组并初始设定等于票息
        cashflow[-1] = M * (1 + C/m1)# 现金流数组最后的元素调整为票息与本金之和
        weight = cashflow * np.exp(-r * t) / price# 计算时间的权重
        Macaulay_duration = np.sum(t * weight)# 计算带票息债券的麦考利久期
    Modified_duration = Macaulay_duration / (1 + y / m2)# 计算债券的修正久期
    return Modified_duration

# 一个案例
# 沿用上例“09附息国债11”债券信息，计算2020年6月12日该债券的修正久期，同时假定每年复利2次的债券到期收益率上升5个基点，分别运用债券的修正久期和债券定价公式计算债券的最新价格
# 第1步：计算债券的修正久期并用自定义函数Rm将连续复利的到期收益率2.4%转换为每年复利2次的到期收益率
# 计算等价的每年复利2次的到期收益率
y1_TB0911 = Rm(Rc = y_TB0911, m = m_TB0911)
print("计算09附息国债11每年复利2次的到期收益率", round(y1_TB0911, 6))

# 计算修正久期
D2_TB0911 = Mod_Duration(C = C_TB0911, M = par, m1 = m_TB0911, m2 = m_TB0911, y = y1_TB0911, t = Tlist_TB0911)
print("2020年6月12日09附息国债11的修正久期", round(D2_TB0911, 4))

# 第2步：计算当每年复利2次的到期收益率上升5个基点，即从2.4145%增加至2.4150%时债券价格的变化金额
# 用修正久期计算债券价格变化
price_change2 = -D2_TB0911 * price_before * y_change
print("用修正久期计算09附息国债11价格变化", round(price_change2, 4))

# 用修正久期近似计算到期收益率变化后的债券价格
price_new3 = price_before + price_change2
print("用修正久期近似计算到期收益率变化后的09附息国债11价格", round(price_new3, 4))

# 第3步：用债券定价公式计算精确的结果，使用自定义函数Bondprice_onediscount（此函数只有在债券到期收益率为连续复利时才可用）和自定义函数Rc，将每年复利2次的利率转换为连续复利的利率
# 计算等价的连续复利到期收益率
yc_TB0911 = Rc(Rm = y1_TB0911 + y_change, m = m_TB0911)
print("计算09附息国债11新的连续复利到期收益率", round(yc_TB0911, 6))

# 精确计算到期收益率（每年复利2次）变化后的债券价格
price_new4 = Bondprice_onediscount(C = C_TB0911, M = par, m = m_TB0911, y = yc_TB0911, t = Tlist_TB0911)
print("精确计算每年复利2次的到期收益率变化后的09附息国债11价格", round(price_new4, 4))



# 美元久期
# 运用Python自定义一个计算债券美元久期的函数
def Dollar_Duration(C, M, m1, m2, y, t):
    """
    定义一个计算债券美元久期的函数
    C：代表债券的票面利率
    M：代表债券的面值
    m1：代表债券票息每年支付的频次
    m2：代表债券到期收益率每年复利频次，通常m2等于m1
    y：代表每年复利m2次的到期收益率
    t：代表定价日至后续每一期现金流支付日的期限长度，用数组格式输入；零息债券可直接输入数字
    """
    r = m2 * np.log(1 + y / m2)# 计算等价的连续复利到期收益率
    if C == 0:# 针对零息债券
        price = M * np.exp(-r * t)# 计算零息债券的价格
        Macaulay_D = t# 计算零息债券的麦考利久期
    else:# 针对带票息债券
        coupon = np.ones_like(t) * M * C / m1# 创建每一期票息金额的数组
        NPV_coupon = np.sum(coupon * np.exp(-r * t))# 计算每一期票息在定价日的现值之和
        NPV_par = M * np.exp(-r * t[-1])# 计算本金在定价日的现值
        price = NPV_coupon + NPV_par# 计算定价日的债券价格
        cashflow = coupon# 先将现金流设定等于票息
        cashflow[-1] = M * (1 + C / m1)# 数组最后的元素等于票息与本金之和
        weight = cashflow * np.exp(-r * t) / price# 计算时间的权重
        Macaulay_D = np.sum(t * weight)# 计算带票息债券的麦考利久期
    Modified_D = Macaulay_D / (1 + y / m2)# 计算债券的修正久期
    Dollar_D = price * Modified_D# 计算债券的美元久期
    return Dollar_D

# 一个案例：沿用上例“09附息国债11”债券信息，计算2020年6月12日该债券的美元久期
# 计算美元久期
D3_TB0911 = Dollar_Duration(C = C_TB0911, M = par, m1 = m_TB0911, m2 = m_TB0911, y = y1_TB0911, t = Tlist_TB0911)
print("2020年6月12日09附息国债11的美元久期", round(D3_TB0911, 2))
# 通过美元久期可算出债券的基点价值（即当债券到期收益率变动1个基点时债券价格的变化金额）



# 衡量债券利率风险的非线性指标——凸性
# 债券的久期只适用于收益率变化很小的情形
# 当收益率发生大的变化时（如100个基点）
# 债券到期收益率变化100个基点
y_newchange = 0.01

# 上升100个基点后的债券到期收益率
y_new = y_TB0911 + y_newchange

# 精确计算到期收益率变化后的债券价格
price_new5 = Bondprice_onediscount(C = C_TB0911, M = par, m = m_TB0911, y = y_new, t = Tlist_TB0911)
print("精确计算到期收益率上升100个基点后的09附息国债11价格", round(price_new5, 4))
# 运用债券定价公式得出精确债券价格为100.9676元，与使用麦考利久期计算得到的结果相差0.076元，差异非常大



# 凸性的表达式
# 凸性的实质就是债券支付现金流时刻ti平方的加权平均值，而权重与计算久期的权重一致，即在ti时刻债券支付的现金流现值与债券价格的比率
def Convexity(C, M, m, y, t):
    """
    定义一个计算债券凸性的函数
    C：代表债券的票面利率
    M：代表债券的面值
    m：代表债券票息每年支付的频次
    y：代表债券的到期收益率（连续复利）
    t：代表定价日至后续每一期现金流支付日的期限长度，用数组格式输入；零息债券可直接输入数字
    """
    if C == 0:# 针对零息债券
        convexity = pow(t, 2)# 计算零息债券的凸性
    else:# 针对带票息债券
        coupon = np.ones_like(t) * M * C / m# 创建每一期票息金额的数组
        NPV_coupon = np.sum(coupon * np.exp(-y * t))# 计算每一期票息在定价日的现值之和
        NPV_par = M * np.exp(-y * t[-1])# 计算本金在定价日的现值
        price = NPV_coupon + NPV_par# 计算定价日的债券价格
        cashflow = coupon# 先将现金流设定等于票息
        cashflow[-1] = M * (1 + C / m)# 数组最后的元素等于票息与本金之和
        weight = cashflow * np.exp(-y * t) / price# 计算每期现金流时间的权重
        convexity = np.sum(pow(t, 2) * weight)# 计算带票息债券的凸性
    return convexity

# 沿用前例“09附息国债11”债券信息，计算2020年6月12日该债券的凸性
# 计算债券凸性
Convexity_TB0911 = Convexity(C = C_TB0911, M = par, m = m_TB0911, y = y_TB0911, t = Tlist_TB0911)
print("2020年6月12日09附息国债11的凸性", round(Convexity_TB0911, 4))



# 凸性的作用
# 重要的关系式
# 沿用前例“09附息国债11”债券信息，假定2020年6月12日“09附息国债11”连续复利的债券到期收益率上升100个基点，计算债券的最新价格
# 通过Python自定义一个运用麦考利久期和凸性计算债券价格变化金额的函数
def Bondprice_change(B, D, C, y_chg):
    """
    定义一个运用麦考利久期和凸性计算债券价格变化金额的函数
    B：代表到期收益率变化之前的债券价格
    D：代表债券的麦考利久期
    C：代表债券的凸性
    y_chg：代表债券到期收益率的变化金额
    """
    price_change1 = -D * B * y_chg# 根据麦考利久期计算债券价格变化金额
    price_change2 = 0.5 * C * B * pow(y_chg, 2)# 根据凸性计算债券价格变化金额
    price_change = price_change1 + price_change2# 考虑麦考利久期和凸性的债券价格变化金额
    return price_change

# 计算债券价格变化金额
price_change3 = Bondprice_change(B = price_before, D = D1_TB0911, C = Convexity_TB0911, y_chg = y_newchange)
print("考虑麦考利久期和凸性之后的09附息国债11价格变化", round(price_change3, 4))

# 考虑麦考利久期和凸性的债券新价格
price_new6 = price_before + price_change3
print("考虑麦考利久期和凸性之后的09附息国债11最新价格", round(price_new6, 4))



# 凸性对债券价格修正效应的可视化
# 沿用上例“09附息国债11”债券信息，同时假定2020年6月12日连续复利的债券到期收益率变化金额是在[-1.5%, 1.5%]区间取等差数列，
# 分别计算仅考虑麦考利久期，考虑麦考利久期和凸性所对应的债券的新价格，并对比债券定价模型所得价格，测算相关的差异结果并可视化
# 第1步：基于不同的债券到期收益率，分别计算仅考虑麦考利久期，考虑麦考利久期和凸性所对应的债券的新价格
# 创建到期收益率变化金额的等差数列
y_change_list = np.linspace(-0.015, 0.015, 200)

# 变化后的到期收益率
y_new_list = y_TB0911 + y_change_list

# 仅用麦考利久期计算债券价格变化金额
price_change_list1 = -D1_TB0911 * price_before * y_change_list

# 仅用麦考利久期计算债券的新价格
price_new_list1 = price_change_list1 + price_before

# 用麦考利久期和凸性计算债券价格变化金额
price_change_list2 = Bondprice_change(B = price_before, D = D1_TB0911, C = Convexity_TB0911, y_chg = y_change_list)

# 用麦考利久期和凸性计算债券的新价格
price_new_list2 = price_change_list2 + price_before

# 创建存放债券定价模型计算的债券新价格初始数据
price_new_list3 = np.ones_like(y_new_list)

# 运用for语句
for i in range(len(y_new_list)):
    price_new_list3[i] = Bondprice_onediscount(C = C_TB0911, M = par, m = m_TB0911, y = y_new_list[i], t = Tlist_TB0911)# 债券定价模型计算债券新价格

# 第2步：以债券定价模型计算得到的债券新价格作为基准价格，测算仅用麦考利久期计算得到的债券新价格与基准价格之间的差异，
# 同时测算用麦考利久期和凸性计算得到的债券新价格与基准价格之间的差异，并绘制相应的图形
# 仅用麦考利久期计算得到的债券新价格与债券定价模型得到的债券新价格之间的差异
price_diff_list1 = price_new_list1 - price_new_list3

# 用麦考利久期和凸性计算得到的债券新价格与债券定价模型得到的债券新价格之间的差异
price_diff_list2 = price_new_list2 - price_new_list3

plt.figure(figsize = (9, 6))
plt.plot(y_change_list, price_diff_list1, "b-", label = u"仅考虑麦考利久期", lw = 2.5)
plt.plot(y_change_list, price_diff_list2, "m-", label = u"考虑麦考利久期和凸性", lw = 2.5)
plt.xticks(fontsize = 13)
plt.xlabel(u"债券到期收益率的变化", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylabel(u"与债券定价模型之间的债券价格差异", fontsize = 13)
plt.title(u"凸性对债券价格的修正效应", fontsize = 13)
plt.legend(fontsize = 13)
plt.grid()
plt.show()



# 测度债券的信用风险
# 运用Python自定义一个通过债券的到期收益率计算得到债券违约概率（连续复利）的函数
def default_prob(y1, y2, R, T):
    """
    定义通过债券到期收益率计算连续复利违约概率的函数
    y1：代表无风险零息利率，并且是连续复利
    y2：代表存在信用风险的债券到期收益率，并且是连续复利
    R：代表债券的违约回收率
    T：代表债券的期限（年）
    """
    A = (np.exp(-y2 * T) - R * np.exp(-y1 * T)) / (1 - R)# 式7-22中的圆括号内的表达式
    prob = -np.log(A) / T - y1# 计算连续复利的违约概率
    return prob

# 一个案例（P239）
# 假定2020年9月1日，计算2只不同信用评级债券连续复利的违约概率，无风险零息利率为参考国债的到期收益率，当天3年期和5年期的无风险零息利率（连续复利）分别为2.922%和2.9811%
# 16宜章养老债的剩余期限
T_yz = 3

# 14冀建投的剩余期限
T_jj = 5

# 16宜章养老债的到期收益率
y_yz = 0.073611

# 14冀建投的到期收益率
y_jj = 0.042471

# 16宜章养老债的违约回收率
R_yz = 0.381

# 14冀建投的违约回收率
R_jj = 0.696

# 3年期无风险零息利率
rate_3y = 0.02922

# 5年期无风险零息利率
rate_5y = 0.029811

# 16宜章养老债的违约概率
default_yz = default_prob(y1 = rate_3y, y2 = y_yz, R = R_yz, T = T_yz)

# 14冀建投的违约概率
default_jj = default_prob(y1 = rate_5y, y2 = y_jj, R = R_jj, T = T_jj)

print("16宜章养老债连续复利的违约概率", round(default_yz, 4))
print("14冀建投连续复利的违约概率", round(default_jj, 4))



# 考察债券到期收益率和债券违约回收率对违约概率的影响，以14冀建投为分析对象
# 第1步：对14冀建投的到期收益率取[3%, 6%]区间的等差数列，其他变量保持不变，计算相应的违约概率；
# 然后对该债券的违约回收率取[40%, 80%]区间的等差数列，其他变量不变，计算相应的违约概率
# 14冀建投到期收益率的数组
y_jj_list = np.linspace(0.03, 0.06, 100)

# 计算不同的到期收益率对应的违约概率
default_jj_list1 = default_prob(y1 = rate_5y, y2 = y_jj_list, R = R_jj, T = T_jj)

# 14冀建投违约回收率的数组
R_jj_list = np.linspace(0.4, 0.8, 100)

# 计算不同的违约回收率对应的违约概率
default_jj_list2 = default_prob(y1 = rate_5y, y2 = y_jj, R = R_jj_list, T = T_jj)

# 第2步：将第1步计算的结果可视化
plt.figure(figsize = (11, 6))
plt.subplot(1, 2, 1)# 第1行第1列的子图
plt.plot(y_jj_list, default_jj_list1, "r-", lw = 2.5)
plt.xticks(fontsize = 13)
plt.xlabel(u"债券到期收益率", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylabel(u"违约概率", fontsize = 13, rotation = 90)
plt.title(u"债券到期收益率与违约概率的关系图", fontsize = 14)
plt.grid()
plt.subplot(1, 2, 2)# 第1行第2列的子图
plt.plot(R_jj_list, default_jj_list2, "b-", lw = 2.5)
plt.xticks(fontsize = 13)
plt.xlabel(u"债券违约回收率", fontsize = 13)
plt.yticks(fontsize = 13)
plt.title(u"债券违约回收率与违约概率的关系图", fontsize = 14)
plt.grid()
plt.show()
