# 第四章
# 基本函数
# 导入Matplotlib模块
import matplotlib

# 查看版本信息
matplotlib.__version__

# 导入子模块pyplot
import matplotlib.pyplot as plt

"""
pyplot子模块中的常用绘图函数与参数：
1.  figure函数：定义画面大小（figure(figsize, dpi, facecolor, edgecolor, frameon)）
    figsize参数：设定画面的宽，高，单位英寸，eg：figsize = (9, 6)即宽9英寸，高6英寸
    dpi参数：画面分辨率，即每英寸多少像素，默认为80
    facecolor参数：画面的背景色
    edgecolor参数：画面的边框颜色
    frameon参数：画面是否显示边框，frameon = True显示边框，False不显示，默认显示边框

2.  plot函数：曲线图（plot(x, y, label, format_string)）
    x参数：对应于x轴的数据
    y参数：对应于y轴的数据
    label参数：曲线的标签
    format_string参数：控制曲线的格式（设定颜色、样式、宽度（lwd，单位磅））

3.  subplot函数：子图（subplot(nrows, ncols, index)）
    nrows参数：子图的行数
    ncols参数：子图的列数
    index参数：子图的序号（最大序号为行数与列数的乘积）

4.  hist函数：直方图（hist(x, bins, facecolor, edgecolor)）
    x参数：每个矩形分布所对应的数据，对应x轴
    bins参数：图中的矩形数量，eg：bins = 20即20个矩形
    facecolor参数：矩形的背景色
    edgecolor参数：矩形的边框色

5.  bar函数：垂直条形图（bar(x, height, width)）
    x参数：条形图x坐标的相关数据
    height参数：每个条形图案的高度
    width参数：每个条形图案的宽度（可选）

6.  barh函数：水平条形图（barh(y, width, height)）
    y参数：条形图y坐标的相关数据
    width参数：每个条形图案的宽度
    height参数：每个条形图案的高度（可选）

7.  scatter函数：散点图（scatter(x, y, c, marker)）
    x参数：x变量的数据
    y参数：y变量的数据
    c参数：散点的颜色（默认为蓝色）
    marker参数：散点的样式

8.  pie函数：饼图（pie(x, labels, colors)）
    x参数：饼图中各块饼的占比
    labels参数：饼图中每块饼的标签文字
    colors参数：饼图中各块饼的颜色

9.  axis函数：坐标轴（axis(xmin, xmax, ymin, ymax, 后面的参数可选：off, equal, scaled, tight, image, square)）
    xmin参数：设置x轴刻度的最小值
    xmax参数：设置x轴刻度的最大值
    ymin参数：设置y轴刻度的最小值
    ymax参数：设置y轴刻度的最大值
    off参数：关闭坐标轴的轴线和标签
    equal参数：使用等刻度的坐标轴
    scaled参数：通过尺寸变化平衡坐标轴的刻度
    tight参数：设置限值使所有数据可见
    image参数：使刻度的限值等于数据的限值
    square参数：与scaled参数相似，但强制要求xmax - xmin = ymax - ymin

10. xticks函数：x轴的刻度（xticks(ticks, labels)）
    ticks参数：x轴刻度的列表，若放置空列表即表示禁用xticks
    labels参数：在给定x轴刻度位置的标签

11. xlabel函数：x轴的坐标标签（xlabel(fontsize, rotation)）（输入字符串以输出x轴的坐标标签）
    fontsize参数：通过输入fontsize = 数字，控制标签字体的大小，单位磅
    rotation参数：通过输入rotation = 数字，控制标签的角度，eg：rotation = 30即标签逆时针旋转30°

12. xlim函数：x轴刻度范围（xlim(xmin, xmax)）
    xmin参数：x轴刻度的最小值
    xmax参数：x轴刻度的最大值

13. yticks函数：y轴的刻度（同xticks函数）

14. ylabel函数：y轴的坐标标签（同xlabe函数）

15. ylim函数：y轴刻度范围（同xlim函数）

16. title函数：图形的标题（title(fontsize)）（输入字符串以输出图形的标题）
    fontsize参数：通过输入fontsize = 数字，控制标题字体的大小

17. legend函数：显示图例（legend(loc)）
    loc参数：通过输入loc = 数字，控制图例的位置（0：最佳；1：右上；2：左上；3：左下；4：右下；5：右；6：中左；7：中右；8：中下；9：中上；10：中；空白：自动）

18. annotate函数：添加注释（annotate(s, xy, xytext, arrowprops（参数包括：width，frac，headwidth，headlength，shrink等）)）
    s参数：注释的内容，以字符串形式输入
    xy参数：标注的位置，以xy = (数字1, 数字2)的元组格式输入，数字1对应x轴刻度，数字2对应y轴刻度
    xytext参数：文本的位置，以xytext = (数字1, 数字2)的元组格式输入，数字1，数字2同上
    arrowprops参数：设置箭头的特征，以字典形式输入，参数包括width（箭头宽度，单位磅），frac（箭头头部所占比例），headwidth（箭头底部宽度，单位磅），
                    headlength（箭头头部长度，单位磅），shrink（箭头收缩程度）等

19. grid函数：网格（grid(axis, color, linestyle, linewidth)）（通常不输入参数）
    axis参数：绘制哪一组网格线，axis = x即仅绘制x轴的网格线，axis = y即仅绘制y轴的网格线，axis = both即绘制x、y轴的网格线（默认）
    color参数：网格线颜色
    linestyle参数：网格线的样式
    linewidth参数：网格线的宽度
"""
# 了解更全面的参数信息：help(plt.函数名)

"""
颜色参数：
b：蓝色
g：绿色
r：红色
c：青色
m：品红色
y：黄色
k：黑色
w：白色
"""

"""
样式或标记参数：
-：实线
--：短画线
-.：点实线
:：虚线
.：点
o：圆
v：向下三角
^：向上三角
<：向左三角
>：向右三角
1：倒三角
2：正三角
3：左三角
4：右三角
s：方形
p：五边形
*：星号
h：六角星标记1
H：六角星标记2
+：加号
x：×型
D：菱形
d：细菱形
l：垂直标记
"""

# 让Matplotlib输出的图形显示中文字体
# 从pylab导入子模块mpl
from pylab import mpl

# 以仿宋字体显示中文
mpl.rcParams["font.sans-serif"] = ["FangSong"]

# 在图像中正常显示负号
mpl.rcParams["axes.unicode_minus"] = False

# 导入NumPy模块和pandas模块
import numpy as np
import pandas as pd

# 在新版pandas中需要下述2条代码才能成功注册日期时间转换器并用于Matplotlib的可视化编程
# 导入注册日期时间转换器的函数
from pandas.plotting import register_matplotlib_converters

# 注册日期时间转换器
register_matplotlib_converters()

# 曲线图
# 单一曲线图
# 针对住房按揭贷款，根据等额本息还款规则，可计算得到每月还款的金额和每月还款金额中所含的本金与利息
# A购房者（借款人）向B银行（贷款人）申请本金为800万元、期限为30年的住房按揭贷款，采用等额本息还款规则进行逐月还款，住房按揭贷款的年利率为5%，将计算得到的每月偿还金额、每月偿还本金金额和每月偿还利息金额进行可视化
# 导入numpy_financial模块
import numpy_financial as npf

# 贷款的年利率
r = 0.05

# 贷款的期限（年）
n = 30

# 贷款的本金
principle = 8e6

# 计算每月支付的本息和
pay_month = npf.pmt(rate=r / 12, nper=n * 12, pv=principle, fv=0, when="end")

# 输出结果
print("每月偿还的金额", round(pay_month, 2))

# 生成一个包含每次还款期限的数组
T_list = np.arange(n * 12) + 1

# 计算每月偿还的本金金额
prin_month = npf.ppmt(rate=r / 12, per=T_list, nper=n * 12, pv=principle, fv=0, when="end")

# 计算每月偿还的利息金额
inte_month = npf.ipmt(rate=r / 12, per=T_list, nper=n * 12, pv=principle, fv=0, when="end")

# 创建每月还款金额的数组
pay_month_list = pay_month * np.ones_like(prin_month)

# 设定参数并绘制曲线图
plt.figure(figsize=(9, 6), frameon=False)
plt.plot(T_list, -pay_month_list, "r-", label=u"每月偿还金额", lw=2.5)
plt.plot(T_list, -prin_month, "m--", label=u"每月偿还本金金额", lw=2.5)
plt.plot(T_list, -inte_month, "b--", label=u"每月偿还利息金额", lw=2.5)
plt.xticks(fontsize=14)
plt.xlim(0, 360)
plt.xlabel(u"逐次偿还的期限（月）", fontsize=14)
plt.yticks(fontsize=13)
plt.ylabel(u"金额", fontsize=13)
plt.title(u"等额本息还款规则下每月偿还的金额以及本金与利息", fontsize=14)
plt.legend(loc=0, fontsize=13)
plt.grid()
plt.show()

# 利率在[3%, 7%]之间进行等差取值，计算对应的每月偿还金额，并进行可视化
# 模拟不同的贷款利率
r_list = np.linspace(0.03, 0.07, 100)

# 计算不同贷款利率条件下的每月偿还本息之和
pay_month_list = npf.pmt(rate=r_list / 12, nper=n * 12, pv=principle, fv=0, when="end")

# 设定参数并绘制曲线图
plt.figure(figsize=(9, 6))
plt.plot(r_list, -pay_month_list, "r-", label=u"每月偿还金额", lw=2.5)
plt.plot(r, -pay_month, "o", label=u"贷款利率5%对应的每月偿还金额", lw=2.5)
plt.xticks(fontsize=14)
plt.xlabel(u"贷款利率", fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(u"金额", fontsize=14)
plt.annotate(u"贷款利率等于5%", fontsize=14, xy=(0.05, 43000), xytext=(0.045, 48000),
             arrowprops=dict(facecolor="m", shrink=0.05))
plt.title(u"不同贷款利率与每月偿还金额之间的关系", fontsize=14)
plt.legend(loc=0, fontsize=14)
plt.grid()
plt.show()

# 多图绘制
# 绘制多张图且每张图以子图的形式显示和排布（subplot函数）
# 以深证成指作为对象进行演示
# 导入2018年至2020年期间深证成指每日开盘价、最高价、最低价、收盘价的数据并创建数据框
SZ_Index = pd.read_excel("C:/Users/InsaneHe/desktop/Python/深证成指每日价格数据（2018-2020年）.xlsx", sheet_name="Sheet1",
                         header=0, index_col=0)

# 显示行索引的格式
print(SZ_Index.index)

# 将数据框的行索引转换为Datetime格式
SZ_Index.index = pd.DatetimeIndex(SZ_Index.index)

# 显示更新后的行索引格式
print(SZ_Index.index)
# 可看出数据框行索引的最初格式为object，通过DatetimeIndex函数完成了Datetime格式的转换

# 设定参数并绘制曲线图
plt.figure(figsize=(11, 9))

# 第1张子图
plt.subplot(2, 2, 1)
plt.plot(SZ_Index["开盘价"], "r-", label=u"深证成指开盘价", lw=2.0)
plt.xticks(fontsize=13, rotation=30)
plt.xlabel(u"日期", fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u"价格", fontsize=13)
plt.legend(loc=0, fontsize=13)
plt.grid()

# 第2张子图
plt.subplot(2, 2, 2)
plt.plot(SZ_Index["最高价"], "b-", label=u"深证成指最高价", lw=2.0)
plt.xticks(fontsize=13, rotation=30)
plt.xlabel(u"日期", fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u"价格", fontsize=13)
plt.legend(loc=0, fontsize=13)
plt.grid()

# 第3张子图
plt.subplot(2, 2, 3)
plt.plot(SZ_Index["最低价"], "c-", label=u"深证成指最低价", lw=2.0)
plt.xticks(fontsize=13, rotation=30)
plt.xlabel(u"日期", fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u"价格", fontsize=13)
plt.legend(loc=0, fontsize=13)
plt.grid()

# 第4张子图
plt.subplot(2, 2, 4)
plt.plot(SZ_Index["收盘价"], "k-", label=u"深证成指收盘价", lw=2.0)
plt.xticks(fontsize=13, rotation=30)
plt.xlabel(u"日期", fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u"价格", fontsize=13)
plt.legend(loc=0, fontsize=13)
plt.grid()

# 自动调整子图之间的间距
plt.tight_layout()

# 绘制图像
plt.show()

# 直方图（hist函数）
# 将变量的全部样本数据按照不同区间范围划分为若干组，横坐标表示变量的样本数据的可取数值，纵坐标表示频数
# 使用NumPy获取基于不同统计分布的随机数作为绘制直方图的数据源（正态分布，对数正态分布，卡方分布和贝塔分布）
# 依次从每个分布中抽取10000个样本值，最后以2*2子图的方式呈现
# 导入NumPy的random子模块
import numpy.random as npr

# 随机抽样的次数
I = 10000

# 从均值为0.8，标准差为1.6的正态分布中随机抽样
x_norm = npr.normal(loc=0.8, scale=1.6, size=I)

# 从均值等于0.5，标准差为1.0的对数正态分布中随机抽样
x_logn = npr.lognormal(mean=0.5, sigma=1.0, size=I)

# 从自由度为5的卡方分布中随机抽样
x_chi = npr.chisquare(df=5, size=I)

# 从α为2，β为6的贝塔分布中随机抽样
x_beta = npr.beta(a=2, b=6, size=I)

# 设定参数并绘制直方图
# 第1张子图（正态分布）
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.hist(x_norm, label=u"正态分布的抽样", bins=20, facecolor="y", edgecolor="k")
plt.xticks(fontsize=13)
plt.xlabel(u"样本值", fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u"频数", fontsize=13)
plt.legend(loc=0, fontsize=13)
plt.grid()

# 第2张子图（对数正态分布）
plt.subplot(2, 2, 2)
plt.hist(x_logn, label=u"对数正态分布的抽样", bins=20, facecolor="r", edgecolor="k")
plt.xticks(fontsize=13)
plt.xlabel(u"样本值", fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u"频数", fontsize=13)
plt.legend(loc=0, fontsize=13)
plt.grid()

# 第3张子图（卡方分布）
plt.subplot(2, 2, 3)
plt.hist(x_chi, label=u"卡方分布的抽样", bins=20, facecolor="r", edgecolor="k")
plt.xticks(fontsize=13)
plt.xlabel(u"样本值", fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u"频数", fontsize=13)
plt.legend(loc=0, fontsize=13)
plt.grid()

# 第4张子图（贝塔分布）
plt.subplot(2, 2, 4)
plt.hist(x_beta, label=u"贝塔分布的抽样", bins=20, facecolor="r", edgecolor="k")
plt.xticks(fontsize=13)
plt.xlabel(u"样本值", fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u"频数", fontsize=13)
plt.legend(loc=0, fontsize=13)
plt.grid()

# 绘制
plt.show()

# 多个样本的直方图
# 为便于展示，比较多组样本值的分布情况，将不同组的样本数据放在一张直方图内对比展示：方法一（堆叠），方法二（并排）
# 堆叠展示
# 从外部导入2019年至2020年期间上证综指，深证成指的日涨跌幅数据
SH_SZ_Index = pd.read_excel("C:/Users/InsaneHe/desktop/Python/上证综指和深证成指的日涨跌幅数据（2019-2020年）.xlsx",
                            sheet_name="Sheet1",
                            header=0, index_col=0)

# 将数据框格式转为数组格式
SH_SZ_Index = np.array(SH_SZ_Index)

# 设定参数并绘制直方图
plt.figure(figsize=(9, 6))
plt.hist(SH_SZ_Index, label=[u"上证综指日涨跌幅", u"深证成指日涨跌幅"], stacked=True, edgecolor="k",
         bins=30)  # 两组数据堆叠显示
plt.xticks(fontsize=13)
plt.xlabel(u"日涨跌幅", fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u"频数", fontsize=13)
plt.title(u"上证综指和深证成指日涨跌幅堆叠的直方图", fontsize=13)
plt.legend(loc=0, fontsize=13)
plt.grid()
plt.show()
# 无论是上证综指还是深证成指，日涨跌幅数据都在[-4%, 4%]区间，同时，最大的日跌幅在-8%附近，最大的日涨幅未能超过6%

# 并排显示
# 沿用上面的两组样本值
plt.figure(figsize=(9, 6))
plt.hist(SH_SZ_Index, label=[u"上证综指日涨跌幅", u"深证成指日涨跌幅"], edgecolor="k", bins=30)  # 两组数据并排显示
plt.xticks(fontsize=13)
plt.xlabel(u"日涨跌幅", fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u"频数", fontsize=13)
plt.title(u"上证综指和深证成指日涨跌幅堆叠的直方图", fontsize=13)
plt.legend(loc=0, fontsize=13)
plt.grid()
plt.show()
# 可看出，相邻的两个矩形均来自不同的数据组，故可以很方便地观察并比较不同数据组的分布情况，目测得：日涨跌幅为0附近，上证综指样本数量明显多于深证成指，
# 随着日涨跌幅逐渐远离0，深证成指的样本数据量多于上证综指。故深证成指的样本数据在分布上比上证综指更分散，其风险更高


# 条形图（比较不同金融资产的收益率，对比不同时期的交易量）
# 垂直条形图
# 针对前述的2020年5月25日至29日的4只A股股票的日涨跌幅数据创建2020年5月25日、26日、28日、29日这4只股票的涨跌幅的垂直条形图并以2*2的子图呈现
# 创建数组
R_array = np.array([[-0.035099, 0.017230, -0.003450, -0.024551, 0.039368],
                    [-0.013892, 0.024334, -0.033758, 0.014622, 0.000128],
                    [0.005848, -0.002907, 0.005831, 0.005797, -0.005764],
                    [0.021242, 0.002133, -0.029803, -0.002743, -0.014301]])

# 输入交易日
date = ["2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29"]

# 输入股票名称
name = ["中国卫星", "中国软件", "中国银行", "上汽集团"]

# 创建数据框
R_dataframe = pd.DataFrame(data=R_array.T, index=date, columns=name)

# 设定参数并绘制直方图
plt.figure(figsize=(12, 10))

# 第1张子图
plt.subplot(2, 2, 1)
plt.bar(x=R_dataframe.columns, height=R_dataframe.iloc[0], width=0.5, label=u"2020年5月25日涨跌幅", facecolor="y")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u"涨跌幅", fontsize=13)
plt.legend(loc=0, fontsize=13)
plt.grid()

# 第2张子图
plt.subplot(2, 2, 2, sharex=plt.subplot(2, 2, 1), sharey=plt.subplot(2, 2, 1))  # 与第1张子图的x轴和y轴相同
plt.bar(x=R_dataframe.columns, height=R_dataframe.iloc[1], width=0.5, label=u"2020年5月26日涨跌幅", facecolor="c")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(loc=0, fontsize=13)
plt.grid()

# 第3张子图
plt.subplot(2, 2, 3, sharex=plt.subplot(2, 2, 1), sharey=plt.subplot(2, 2, 1))  # 与第1张子图的x轴和y轴相同
plt.bar(x=R_dataframe.columns, height=R_dataframe.iloc[3], width=0.5, label=u"2020年5月28日涨跌幅", facecolor="b")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(loc=0, fontsize=13)
plt.grid()

# 第4张子图
plt.subplot(2, 2, 4, sharex=plt.subplot(2, 2, 1), sharey=plt.subplot(2, 2, 1))  # 与第1张子图的x轴和y轴相同
plt.bar(x=R_dataframe.columns, height=R_dataframe.iloc[4], width=0.5, label=u"2020年5月29日涨跌幅", facecolor="g")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(loc=0, fontsize=13)
plt.grid()

# 绘制
plt.show()

# 水平条形图（barh函数）
# 将2020年5月26日与27日这两个交易日4只股票的涨跌幅放置在一张水平条形图中进行展示
plt.figure(figsize=(9, 6))
plt.barh(y=R_dataframe.columns, width=R_dataframe.iloc[1], height=0.5, label=u"2020年5月26日涨跌幅")
plt.barh(y=R_dataframe.columns, width=R_dataframe.iloc[2], height=0.5, label=u"2020年5月27日涨跌幅")
plt.xticks(fontsize=13)
plt.xlabel(u"涨跌幅", fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u"水平条形图可视化股票的涨跌幅数据", fontsize=13)
plt.legend(loc=1, fontsize=13)
plt.grid()
plt.show()

# 综合条形图与折线图的双轴图（subplots函数和twinx函数）
# 条形图：描述变量的金额，对应左侧y轴；折线图：刻画变量的变化情况（增长率），对应右侧y轴
# subplots函数创建含figure和axes对象的元组，twinx函数创建一个右侧纵坐标
# 导入2019年至2020年我国广义货币供应量M2每月余额和每月同比增长率的数据并创建一个数据框
M2 = pd.read_excel("C:/Users/InsaneHe/desktop/Python/我国广义货币供应量M2的数据（2019-2020年）.xlsx", sheet_name="Sheet1",
                   header=0, index_col=0)

# 运用左侧纵坐标绘制图形
fig, ax1 = plt.subplots(figsize=(9, 6))
plt.bar(x=M2.index, height=M2.iloc[:, 0], width=20, color="y", label=u"M2每月余额")
plt.xticks(fontsize=13, rotation=90)
plt.xlabel(u"日期", fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(0, 250)
plt.ylabel(u"金额（万亿元）", fontsize=13)
plt.legend(loc=2, fontsize=13)  # 图例的位置设置在左上方

# 运用右侧纵坐标绘制图形
ax2 = ax1.twinx()
plt.plot(M2.iloc[:, -1], label=u"M2每月同比增长率", lw=2.5)
plt.yticks(fontsize=13)
plt.ylim(0, 0.13)
plt.ylabel(u"增长率", fontsize=13)
plt.title(u"广义货币供应量M2每月余额和每月同比增长率", fontsize=13)
plt.legend(loc=1, fontsize=13)  # 图例的位置设置在右上方
plt.grid()
plt.show()

# 散点图（scatter函数）
# 将两个变量的样本值显示为一组点，样本值由点在图中的位置表示，用于识别两个变量之间的线性相关性或用于观察它们之间的关系以发现趋势
# 散点越向一条直线靠拢，两个变量之间的线性相关性越高，反之越低
# 分析2016年至2020年期间中国工商银行与中国建设银行这两只A股股票的周涨跌幅
ICBC_CCB = pd.read_excel("C:/Users/InsaneHe/desktop/Python/工商银行与建设银行A股周涨跌幅数据（2016-2020年）.xlsx",
                         sheet_name="Sheet1",
                         header=0, index_col=0)

# 查看数据框的描述性统计
print(ICBC_CCB.describe())

# 工商银行与建设银行周涨跌幅的相关系数
print(ICBC_CCB.corr())
# 可看出就数据的描述性统计而言，工商银行与建设银行的周涨跌幅，无论均值、标准差还是主要的分位数都比较接近，
# 同时相关系数超过0.85，可初步判断为两只股票在周涨跌幅上具有高度的线性相关性，可推出在散点图中的散点是比较靠近于一条直线的

# 绘制工商银行与建设银行周涨跌幅的散点图
plt.figure(figsize=(9, 6))
plt.scatter(x=ICBC_CCB["工商银行"], y=ICBC_CCB["建设银行"], c="r", marker="o")
plt.xticks(fontsize=13)
plt.xlabel(u"工商银行周涨跌幅", fontsize=13)
plt.yticks(fontsize=13)
plt.ylabel(u"建设银行周涨跌幅", fontsize=13)
plt.title(u"工商银行与建设银行周涨跌幅的散点图", fontsize=13)
plt.grid()
plt.show()
# 印证了前推结果，即工商银行与建设银行的周涨跌幅解百纳处于一条直线附近，但是相关关系不等于因果关系


# 饼图（pie函数）
# 用于计算变量的若干样本值占总样本值的比重
# 以国际货币基金组织特别提款权中不同币种的权重为例
# 创建存放币种名称的列表
currency = ["美元", "欧元", "人民币", "日元", "英镑"]

# 创建存放不同币种权重的列表
perc = [0.4173, 0.3093, 0.1092, 0.0833, 0.0809]

# 设定参数并绘制图像
plt.figure(figsize=(9, 7))
plt.pie(x=perc, labels=currency, textprops={"fontsize": 13})  # textprops参数用于控制饼图中标签的字体大小，此处设定为13磅

# 使饼图是一个圆形
plt.axis("equal")
plt.title(u"特别提款权中不同币种的权重", fontsize=13)

# 图例在左上方
plt.legend(loc=2, fontsize=13)
plt.show()

# 雷达图
# 以二维图的形式在从中心点开始向外延伸的数轴上表示3个或更多个变量数据的图形
"""
第一步：输入准备好的参数数据，
使用NumPy的linspace函数：将整个圆形按需要显示的指标数量进行均匀切分（eg：需要显示4个指标，将圆形均匀切分为4个部分）
同时，使用NumPy的concatenate函数：将相关数组进行首位拼接以实现图形的闭合

第二步：绘制
使用Matplotlib的子模块pyplot的polar函数：绘制雷达图的坐标系
使用thetagrids函数：输入图形中涉及的指标名称
"""

# 以国内A股上市保险公司相关财务指标作为对象具体演示雷达图的绘制，展示中国太保在这些指标中的排名情况（指标等见P126）
# 创建存放公司名称的列表
company = ["中国人寿", "中国人保", "中国太保", "中国平安", "新华保险"]

# 创建存放指标名称的列表
indicator = ["营业收入增长率", "净利润增长率", "净资产收益率", "偿付能力充足率"]

# 创建存放中国太保各指标排名的数组
ranking = np.array([5, 4, 3, 2])

# 公司的数量
N_company = len(company)

# 指标的数量
N_indicator = len(indicator)

# 在中国太保各项指标排名的数组末尾增加一个该数组的首位数字，以实现绘图的闭合
ranking_new = np.concatenate([ranking, [ranking[0]]])

# 将圆形按照指标数量进行均匀切分
angles = np.linspace(0, 2 * np.pi, N_indicator, endpoint=False)

# 在已创建的angles数组的末尾增加一个该数组的首位数字，以实现绘图的闭合
angles_new = np.concatenate([angles, [angles[0]]])

# 设定参数以绘制图像
plt.figure(figsize=(8, 8))

# 绘制雷达图
plt.polar(angles_new, ranking_new, "--")

# 绘制图形的指标名称
plt.thetagrids(angles_new * 180 / np.pi, indicator, fontsize=13)
plt.ylim(0, 5)

# 刻度按照公司数量设置
plt.yticks(range(N_company + 1), fontsize=13)

# 对图中相关部分用颜色填充
plt.fill(angles_new, ranking_new, facecolor="r", alpha=0.3)
plt.title(u"中国太保各项指标在5家A股上市保险公司中的排名", fontsize=13)
plt.show()

# K线图（plot函数和mplfinance模块）
# 当收盘价高于开盘价时：K线称为阳线，当收盘价低于开盘价时：K线称为阴线
# 导入mplfinance模块
import mplfinance as mpf

# 查看版本信息
print(mpf.__version__)

"""
plot函数的主要参数及输入方式
1.data参数：表示输入绘制图形的数据，数据需要以数据框格式存放，且数据框需满足：1.行索引必须为Datatime格式；2.列名必须依次用Open, High, 
Low, Volumn等英文字母表示，分别代表开盘价、最高价、最低价、收盘价、交易量等
2.type参数：ohlc：条形图；candle：蜡烛图；line：折线图；renko：砖形图；pnf：OX图或点数图
3.mav参数：均线，可生成一条或若干条，输入mav = 5：生成5日均线（以日K线为例）；输入mav = (5, 10)就表示分别生成5日均线和10日均线
4.volume参数：绘制交易量，输入volume = True就表示绘制交易量；不输入或输入volume = False：不绘制交易量
5.figratio参数：定义画面的尺寸，如输入figratio = (9, 6)代表宽9英寸，高6英寸
6.style参数：设定K线图的图案风格，style = "classic"：经典风格（阳线：白色，阴线：黑色）；
也可调用mplfinance的函数make_marketcolors和make_mpf_style自定义阳线，阴线等图案颜色
7.ylabel参数：y轴的坐标标签
8.ylabel_lower参数：对应绘制交易量图形的y轴坐标标签
"""

# 以2020年第3季度上证综指每个交易日价格和成交额的部分数据，用mplfinance绘制K线图，分两步完成
# 第1步：绘制K线图并采用经典的图案风格，同时需要在图形中绘制5日均线
SH_Index = pd.read_excel("C:/Users/InsaneHe/desktop/Python/2020年第3季度上证综指的日交易数据.xlsx", sheet_name="Sheet1",
                         header=0, index_col=0)

# 数据框的行索引转换为Datatime格式
SH_Index.index = pd.DatetimeIndex(SH_Index.index)

# 显示数据框的列名
print(SH_Index.columns)

# 将数据框的列名调整为英文
SH_Index = SH_Index.rename(columns={"开盘价": "Open", "最高价": "High", "最低价": "Low", "收盘价": "Close",
                                    "成交额（万亿元）": "Volume"})

# 绘制经典风格的K线图
mpf.plot(data=SH_Index, type="candle", mav=5, volume=True, figratio=(9, 7), style="classic", ylabel="price",
         ylabel_lower="volume(trillion)")
# 经典风格的K线图分为两个部分，同时y轴的刻度和标签默认是在右侧。图形上半部分为价格走势图，其中阳线用白色表示，阴线用黑色表示，曲线代表5日均线
# 下半部分刻画了每日的交易情况

# 第2步：绘制K线图并且采用阳线用红色表示、阴线用绿色表示的图案风格，同时需要在图形中绘制5日均线和10日均线
# 设置阳线用红色表示，阴线用绿色表示
color = mpf.make_marketcolors(up="r", down="g")

# 运用make_mpf_style函数
style_color = mpf.make_mpf_style(marketcolors=color)

# 绘制自定义的K线图
mpf.plot(data=SH_Index, type="candle", mav=(5, 10), volume=True, figratio=(9, 6), style=style_color,
         ylabel="price", ylabel_lower="volume(trillion)")
