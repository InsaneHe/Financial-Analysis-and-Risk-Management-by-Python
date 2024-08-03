# 第十章
# 商品期货合约的介绍
# 绘制2020年8月到期的黄金期货AU2008合约收盘价、结算价格（“结算价”）、持仓量和成交额的交易日数据绘制走势图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["FangSong"]
mpl.rcParams["axes.unicode_minus"] = False
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# 导入黄金期货AU2008合约数据
data_AU2008 = pd.read_excel("D:/Python/商品期货/黄金期货AU2008合约.xlsx", sheet_name = "Sheet1", header = 0, index_col = 0)

# 可视化
data_AU2008.plot(figsize = (10, 9), subplots = True, layout = (2, 2), grid = True, fontsize = 13)

# 第1张子图
plt.subplot(2, 2, 1)

# 增加第1张子图纵坐标标签
plt.ylabel(u"金额或数量", fontsize = 11, position = (0, 0))
# 可以看出：无论是成交额还是持仓量的走势均呈现“倒U”形，即合约在上市初期以及临近到期日成交额和持仓量均较低，说明合约处于不活跃期；
# 合约在其他期间成交额和持仓量则较高，说明合约处于活跃期



# 股指期货合约的介绍
# TODO: 导入沪深300指数期货IF2009合约数据
data_IF2009 = pd.read_excel("D:/Python/股指期货/沪深300指数期货IF2009合约.xlsx", sheet_name = "Sheet1", header = 0,
                            index_col = 0)

data_IF2009.plot(figsize = (10, 9), subplots = True, layout = (2, 2), grid = True, fontsize = 13)

plt.subplot(2, 2, 1)

plt.ylabel(u"金额或数量", fontsize = 11, position = (0, 0))
# 可以看出：无论是收盘价还是结算价均处于上升通道中，合约成交额和持仓量的走势与黄金期货AU2008合约类似，依然呈现倒U形，
# 特别是持仓量在合约到期日降至0，这是因为股指期货合约采用现金交割模式



# 国债期货合约的介绍
# TODO: 导入10年期国债期货T2009合约数据
data_T2009 = pd.read_excel("D:/Python/国债期货/10年期国债期货T2009合约.xlsx", sheet_name = "Sheet1", header = 0,
                            index_col = 0)

data_T2009.plot(figsize = (10, 9), subplots = True, layout = (2, 2), grid = True, fontsize = 13)

plt.subplot(2, 2, 1)

plt.ylabel(u"金额或数量", fontsize = 11, position = (0, 0))
# 可以看出：收盘价、结算价、成交额和持仓量的走势均呈现倒U形
