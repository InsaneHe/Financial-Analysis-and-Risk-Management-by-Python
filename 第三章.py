# 第三章
# pandas的数据结构分为两类：序列和数据框
# 序列（一个类似于一维数组的数据结构，由两部分组成：索引和对应的数值，且两部分的长度必须一致）
# 创建序列（Series函数：Series(data, index)，data：用于输入相关的数据或变量，index：用于输入索引）
# 使用2.1节例2-1
# 导入模块
import pandas as pd

# 查看版本
print(pd.__version__)

# 对表3-1中2020年5月25日的日涨跌幅数据创建序列
# 输入股票名称
name = ["中国卫星", "中国软件", "中国银行", "上汽集团"]

# 输入2020年5月25日的日涨跌幅数据
list_May25 = [-0.035099, -0.013892, 0.005848, 0.021242]

# 创建序列
series_May25 = pd.Series(data = list_May25, index = name)

# 输出结果
print(series_May25)

# 通过return_array1数组创建2020年5月27日的日涨跌幅数据的序列
# 创建数组
return_array1 = np.array([[-0.035099, 0.017230, -0.003450, -0.024551, 0.039368],
                         [-0.013892, 0.024334, -0.033758, 0.014622, 0.000128],
                         [0.005848, -0.002907, 0.005831, 0.005797, -0.005764],
                         [0.021242, 0.002133, -0.029803, -0.002743, -0.014301]])

# 通过数组创建序列
series_Mar27 = pd.Series(data = return_array1[:, 2], index = name)

# 输出结果
print(series_Mar27)

# 运用日期作为索引，并且以中国银行在2020年5月25日至29日期间的日涨跌幅数据作为对象创建序列
# 输入交易日
date = ["2020-05-25", "2020-05-26", "2020-05-27", "2020-05-28", "2020-05-29"]

# 创建中国银行序列
series_BOC = pd.Series(data = return_array1[2, :], index = date)

# 输出结果
print(series_BOC)

# 由上可得：序列只能有2列，一列为索引列，另一列为数值列



# 数据框（可以存放更多列，由3部分组成：行索引，列名和数值）
# 创建数据框（DataFrame函数：DataFrame(data, index, columns)，data：输入相关数据或变量，index：输入行索引，columns：输入列名）
# 使用例3-1信息，将日期作为行索引，将股票名称作为列名，创建数据框
# 创建数据框
return_dataframe = pd.DataFrame(data = return_array1.T, index = date, columns = name)

# 输出结果
print(return_dataframe)

# 可以将数据框以excel，csv或txt等格式导出，具体运用导出函数to_excel和to_csv，且需要在函数中添加导出文件存放路径和带个是的文件名
# 将return_dataframe依次以Excel，CSV和TXT格式导出并存放在桌面
# 以Excel格式导出
return_dataframe.to_excel("C:/Users/InsaneHe/desktop/四只股票涨跌幅数据.xlsx")

# 以CSV格式导出
return_dataframe.to_csv("C:/Users/InsaneHe/desktop/四只股票涨跌幅数据.csv", encoding = "utf_8_sig")
# 导出CSV文件时加上encoding参数，设置为utf_8_sig以用UTF-8编码，这样不会出现乱码

# 以TXT格式导出
return_dataframe.to_csv("C:/Users/InsaneHe/desktop/四只股票涨跌幅数据.txt")

"""
从外部文件导入数据：
1.导入Excel文件：read_excel("文件的路径", sheetname, header, index_col)
2.导入CSV文件：read_excel("文件的路径或网址", sep, delimiter, header, index_col)
3.导入TXT文件：read_table("文件的路径", index_col, delimiter)
"""

# 导入上证综指2020年每个交易日开盘价、最高价、最低价和收盘价的Excel文件并创建数据框
# 导入外部数据
SH_Index = pd.read_excel("C:/Users/InsaneHe/desktop/Python/上证综指每日交易价格数据（2020年）.xlsx", thousands = ",",
                         sheet_name = "Sheet1", header = 0, index_col = 0)

# 显示开头5行
print(SH_Index.head())

# 显示末尾5行
print(SH_Index.tail())



# 创建序列或数据框的时间序列
# 创建时间序列（date_range函数，date_range(start, end, periods, freq)，start：时间序列的起始时间（以字符串形式输入），end：时间序列的终止时间，
# periods：时间序列包含的时间个数（期数），freq：时间的频次），针对freq进行输入时，也可不输入periods
# 以2021年1月1日作为起始时间，运用date_range函数创建2021年至2022年每个工作日的时间序列
# 创建2021年至2022年每个工作日的时间序列（选择不输入periods）
time1 = pd.date_range(start = "2021-01-01", end = "2022-12-31", freq = "B")
# freq参数中：D：以自然日作为频次，B：以工作日作为频次（不包含双休日的自然日，但是含元旦等公众节假日，其他同），H：以小时作为频次，T/min：以分钟作为频次，S：以秒作为频次
# L/ms：以毫秒作为频次，U：以微秒作为频次，M：以月作为频次（以每月最后一个自然日作为观测时点），BM：以月作为频次（以每月最后一个工作日作为观测时点），
# MS：以月作为频次（以每月第一个自然日作为观测时点），BMS：以月作为频次（以每月第一个工作日作为观测时点）

# 输出结果
print(time1)

# A股每个交易日上午交易从9点30分（不含集合竞价时间）至11点30分
# 创建2021年1月4日这个交易日上午9点30分开始以秒作为频次并且包含7200个时间元素的时间序列（选择输入periods）
time2 = pd.date_range(start = "2021-01-04 09:30:00", periods = 7200, freq = "S")

# 输出结果
print(time2)




# 数据的可视化
# 使用pandas包的plot函数
# 在可视化中中文字体的显示
# 从pylab中导入子模块mpl
from pylab import mpl

# 以仿宋字体显示中文
mpl.rcParams["font.sans-serif"] = ["FangSong"]

# 在图像中正常显示负号“-”
mpl.rcParams["axes.unicode_minus"] = False

# 每次重新启动python都需要将上述部分代码重新输入，否则无法在可视化图形中显示中文字体

"""
黑体：SimHei；微软雅黑：Microsoft YaHei；微软正黑体：Microsoft JhengHei；新宋体：NSimSun；新细明体：PMingLiU；细明体：MingLiU；
标楷体：DFKai-SB；仿宋：FangSong；楷体：KaiTi；仿宋_GB2312：FangSong_GB2312；楷体_GB2312：KaiTi_GB2312
"""



# 数据框可视化的函数与参数
# 主要运用函数plot（plot(kind, subplots, sharex, sharey, layout, figsize, title, grid, fontsize)）
# kind；需要显示的数据类型（line折线图，bar条形图，box箱线图，barh横向条形图，hist柱状图，kde核密度估计图，density同kde，area区域图，pie饼图，scatter散点图，hexbin六边形箱图）
# subplots：判断图像是否存在子图（True为是，False为否），sharex：存在子图时判断子图是否共用x轴刻度及标签，sharey：存在子图时判断子图是否共用y轴刻度及标签
# layout：存在子图时，子图的行列布局情况（行数，列数）
# figsize：输出图形的尺寸大小，eg：figsize = (10, 8)即图形长10英寸，宽8英寸，title：生成图形的标题，用字符串表示，grid：图形中是否需要网格（True为是，默认为否）
# fontsize：设置图形轴刻度的字体大小，eg：fontsize = 13即轴刻度字体为13磅
# 用数据库SH_Index，运用plot函数对该数据框进行可视化
# 可视化
SH_Index.plot(kind = "line", subplots = True, sharex = True, sharey = True, layout = (2, 2), figsize = (11, 9),
              title = u"2020年上证综指每个交易日价格走势图", grid = True, fontsize = 13)
# 由于图形的标题是中文字体，故在输入代码时，除了将中文标题以字符串形式输入外，还要在字符串前加上“u”



# 数据框内部的操作
# 查看数据框的基本性质
# index函数和columns函数
# index函数（查看数据框的行索引名），columns函数（查看数据框的列名）
# 使用数据框SH_Index
# 查看数据框的行索引名
print(SH_Index.index)

# 查看数据框的列名
print(SH_Index.columns)

# shape函数和describe函数
# shape函数（查询一个数据框由多少行、多少列组成），describe函数（对于金融时间序列，查询样本量、均值、标准差、最大值、最小值、分位数等统计指标）
# 使用数据框SH_Index
# 查询数据框的行数和列数
print(SH_Index.shape)
# 以上输出结果为元组，第一个元素为行数（不含列名），第二个元素为列数

# 使用数据框SH_Index
# 查看数据框的基本统计指标
SH_Index.describe()

# count：有多少样本，mean：均值，std：标准差，min：最小值，25%，50%，75%：25%，50%和75%的分位数，max：最大值
# 以上输出结果也是数据框，行索引名为相关统计指标，列名为与数据框SH_Index的列名一致

# 数据框的索引与截取
# 索引（loc函数（通过输入行索引的方式输出对应的数据）和iloc函数（通过输入行号即具体多少行的方式输出对应的数据，且行号0为第1行））
# 以数据框SH_Index为例
# 查看2020年2月18日的数据（loc函数）
SH_Index.loc["2020-02-18"]

# 查看第8行的数据（iloc函数）
SH_Index.iloc[7]

# 一般性截取（针对金融时间序列，截取某个时间区间内相关变量的取值情况）
# 使用数据框SH_Index
# 截取数据框前5行的数据
SH_Index[: 5]

# 截取数据框第8行至第12行的数据
SH_Index[7: 12]

# 截取数据框第17行至第19行以及第2、3列（不包括行索引）的数据
SH_Index.iloc[16: 19, 1: 3]

# 截取2020年5月18日至22日的数据
SH_Index.loc["2020-05-18": "2020-05-22"]

# 条件性截取
# 使用数据框SH_Index，截取收盘价超过3450点的相关数据
SH_Index[SH_Index["收盘"] >= 3450]
# 也可以同时设定多个条件

# 使用数据框SH_Index，截取最高价超过3440点但是最低价低于3380点的相关数据
SH_Index[(SH_Index["高"] >= 3440) & (SH_Index["低"] <= 3380)]
# 不同的选取条件要用英文的圆括号括起来且不同选取条件之间用“&”相连



# 数据框的排序
# 按行索引的大小排序
# 如针对金融时间序列按照时间由近到远（由大到小）或由远到近（由小到大）进行排序（sort_index函数）
# sort_index(ascending)，ascending：默认为True：表示由小到大排序，Flase：表示由大到小排序
# 使用数据框SH_Index，分别输出按照交易日由远到近、由近到远排序的结果
# 按照交易日由远到近排序
SH_Index.sort_index(ascending = True)

# 按照交易日由近到远排序
SH_Index.sort_index(ascending = False)

# 按列名对应的数值大小排序（sort_values函数）
# sort_values(by, asccending)，by = 列名，ascending默认为True：由小到大排序，False：由大到小排序
# 使用数据框SH_Index，分别输出按照开盘价由小到大排序、由大到小排序的结果
# 按照开盘价由小到大排序
SH_Index.sort_values(by = "开盘", ascending = True)

# 按照开盘价由大到小排序
SH_Index.sort_values(by = "开盘", ascending = False)



# 数据框的更改
# 修改行索引与列名（rename函数）
# rename(index, columns)，修改行索引：index = {"原名称": "新名称"}，修改列名：columns = {"原名称": "新名称"}
# 使用数据框SH_Index，将行索引中的交易日“2020-01-02”改为“2020年1月2日”
SH_Index_new = SH_Index.rename(index = {"2020-01-02": "2020年1月2日"})

# 将其中一个列名“收盘价”改为“收盘点位”
SH_Index_new = SH_Index_new.rename(columns = {"收盘价": "收盘点位"})

# 显示列名修改后的前5行
SH_Index_new.head()



# 缺失值的查找
# 由于休假日或停牌，金融时间序列可能出现数据缺失
# 使用isnull函数/isna函数查找数据框中每一列是否存在缺失值（第一步：查找每一列是否存在缺失值，若有进入第二步：精准找出缺失值所在行）
# 导入2020年4月上证综指、道琼斯指数、富时100指数以及日经225指数的日收盘价并创建数据框
Index_global = pd.read_excel("C:/Users/InsaneHe/desktop/Python/全球主要股指2020年4月收盘价数据.xlsx", sheet_name = "Sheet1",
                             header  = 0, index_col = 0)

# 用isnull函数查找每一列是否存在缺失值
Index_global.isnull().any()

# 用isna函数查找每一列是否存在缺失值
Index_global.isna().any()
# 上述代码输出结果中True表示相关的列存在缺失值，即4只股票都存在缺失值

# 用isnull函数查找缺失值所在行
Index_global[Index_global.isnull().values == True]
# 以上代码结果中缺失值用NaN表示，且由于道指和富时在4月10日均休市一天，故重复输出了相同的一行

# 缺失值的处理
"""
对于数据框的缺失值有4种处理方法：
1.直接删除法：用dropna函数直接将存在缺失值的整行数据进行删除
2.零值补齐法：用fillna函数并输入参数value = 0将缺失值赋值为0
3.前值补齐法：用fillna函数并输入参数method = "ffill"用缺失值所在列的前一个非缺失值补齐即向前填充
4.后值补齐法：用fillna函数并输入参数method = "bfill"用缺失值所在列的后一个非缺失值补齐即向后填充
"""

# 用数据框Index_global，依次采用直接删除法，零值补齐法，前值补齐法和后值补齐法对设计的缺失值进行处理
# 直接删除法
Index_dropna = Index_global.dropna()

# 输出结果
print(Index_dropna)
# 4月6日，10日，13日和29日这4个交易日的整行被删除

# 零值补齐法
Index_fillnazero = Index_global.fillna(value = 0)

# 输出结果
print(Index_fillnazero)
# 4月6日，10日，13日和29日这4个交易日的缺失值变为0

# 前值补齐法（较为常见）
Index_ffill = Index_global.fillna(method = "ffill")
# 可在上述代码弃用后用：Index_ffill = Index_global.ffill

# 输出结果
print(Index_ffill)
# 4月6日，10日，13日和29日这4个交易日的缺失值变为所在列的前一个非缺失值

# 后值补齐法
Index_bfill = Index_global.fillna(method = "bfill")
# 可在上述代码弃用后用：Index_bfill = Index_global.bfill

# 输出结果
print(Index_bfill)
# 4月6日，10日，13日和29日这4个交易日的缺失值变为所在列的后一个非缺失值



# 数据框之间的合并
# 合并有两类：1.上下结构（即按行合并），2.左右结构（即按列合并）
# 使用concat/merge/join函数（concat函数用于按行/列合并，merge函数和join函数主要用于按列合并）
# 创建两个新数据框
# 导入2019年上证综指每日交易价格的数据并创建一个新数据框
SH_Index_2019 = pd.read_excel("C:/Users/InsaneHe/desktop/Python/上证综指每日交易价格数据（2019年）.xlsx", thousands = ",",
                              sheet_name = "Sheet1", header = 0, index_col = 0)

# 显示开头5行
SH_Index_2019.head()

# 显示末尾5行
SH_Index_2019.tail()

# 导入2020年上证综指每个交易日的成交额、总市值数据并创建一个新数据框
SH_Index_volume = pd.read_excel()
# 找不到2020年上证综指每个交易日的总市值数据

# 显示开头5行
SH_Index_volume.head()

# 显示末尾5行
SH_Index_volume.tail()



# concat函数的运用
# concat([数据框1, 数据框2, ……, 数据框n], axis = 0/1)
# axis = 0：按行合并，axis = 1：按列合并
# 按行合并
SH_Index_new1 = pd.concat([SH_Index_2019, SH_Index], axis = 0)

# 显示开头5行
SH_Index_new1.head()

# 显示末尾5行
SH_Index_new1.tail()

# 按列合并
# 使用SH_Index_volume和SH_Index数据框
SH_Index_new2 = pd.concat([SH_Index, SH_Index_volume], axis = 1)

# 显示开头5行
SH_Index_new2.head()

# 显示末尾5行
SH_Index_new2.tail()



# merge函数的运用
# merge函数用于对不同数据框按列进行合并（在使用merge函数时，需要明确放置在左侧和右侧的数据框）
# 使用SH_Index和SH_Index_volume
SH_Index_new3 = pd.merge(left = SH_Index, right = SH_Index_volume, left_index = True, right_index = True)
# left_index参数表示是否按照左侧数据框的行索引进行合并，right_index参数表示是否按照右侧数据框的行索引进行合并

# 显示开头5行
SH_Index_new3.head()

# 显示末尾5行
SH_Index_new3.tail()



# join函数的运用（数据框1.join(数据框2, 参数)）
# 使用SH_Index和SH_Index_volume
SH_Index_new4 = Sh_Index.join(SH_Index_volume, on = "日期")
# on参数表示按照某个或某几个索引进行合并，默认为依据两个数据框的行索引进行合并

# 显示开头5行
SH_Index_new4.head()

# 显示末尾5行
SH_Index_new4.tail()



# 数据框的主要统计函数
# 静态统计函数（以SH_Index_new1作为对象）
# 查看最小值（min函数）
SH_Index_new1.min()

# 查看最小值的行索引值（idxmin函数）
SH_Index_new1.idxmin()

# 查看最大值（max函数）
SH_Index_new1.max()

# 查看最大值的行索引值（idxmax函数）
SH_Index_new1.idxmax()

# 查看中位数（median函数）
SH_Index_new1.median()

# 查看分位数（quantile函数）
# 计算5%分位数
SH_Index_new1.quantile(q = 0.05)

# 计算50%分位数
SH_Index_new1.quantile(q = 0.5)

# 计算均值（一阶矩）（mean函数）
SH_Index_new1.mean()

# 计算样本方差（二阶矩）（var函数）
SH_Index_new1.var()

# 计算样本标准差（std函数）
SH_Index_new1.std()

# 计算偏度（三阶矩）（skew函数）
SH_Index_new1.skew()
# 输出结果表明偏度>0，表示上证综指的价格分布具有正偏态（即右偏态），若偏度<0，则表明数据分布具有负偏态（即左偏态）

# 计算峰度（四阶矩）（kurt函数）
SH_Index_new1.kurt()
# 输出结果表明峰度<0，表明相比正态分布，上证综指的价格分布更扁平，若峰度>0，则表明数据分布相比正态分布更陡峭

# 数据框移动（shift函数）
# shift(1)表示数据框的每一行均向下移动1行，shift(2)表示数据框的每一行均向下移动2行……以此类推
# 每行均向下移动1行
SH_Index_shift1 = SH_Index_new1.shift(1)

# 查看前5行
SH_Index_shift1.head()
# 原数据框中的2019年1月2日的数据在新数据框中被移动到2019年1月3日……以此类推，由于原数据框无2019年1月2日更早的数据，故2019年1月2日的数据由NaN表示

# 计算一阶差分（diff函数）（后-前）
SH_Index_diff = SH_Index_new1.diff()

# 查看前5行
SH_Index_diff.head()
# 新数据框中的2019年1月3日的数据等于原数据框中的2019年1月3日数据-2019年1月2日数据……以此类推，由于原数据框无2019年1月2日更早的数据，故2019年1月2日的数据由NaN表示

# 计算百分比变化（pct_change函数）
SH_Index_perc = SH_Index_new1.pct_change()

# 查看前5行
SH_Index_perc.head()

# 求和（sum函数）
# 删除存在缺失值的行
SH_Index_perc = SH_Index_perc.dropna()

# 对百分比变化的数据框求和
SH_Index_perc.sum()

# 累积求和（cumsum函数）
# 对百分比变化的数据框累积求和
SH_Index_cumsum = SH_Index_perc.cumsum()

# 查看前5行
SH_Index_cumsum.head()
# 新数据框中的2019年1月3日数据就是原数据框（SH_Index_perc）该交易日的数据，新数据框中的2019年1月4日数据就是原数据框该交易日的数据加上该交易日之前的数据……以此类推，通过cumsum函数可用依次求出元数据款前1，2，……，n个数的和

# 累积求积（cumprod函数）
# 百分比变化的数据框每个元素均加上1
SH_Index_chag = SH_Index_perc + 1

# 对新数据框累积求积
SH_Index_cumchag = SH_Index_chag.cumprod()

# 查看前5行
SH_Index_cumchag.head()
# cumprod函数和cumsum函数类似，区别在于cumprod函数是依次求出前1，2，……，n个数的积

# 协方差（cov函数）
SH_Index_perc.cov()

# 相关系数（corr函数）
SH_Index_perc.corr()
# 可看出开盘价与最低价之间的相关性最强，相关系数接近0.8，相比之下开盘价和收盘价之间的相关性最弱，相关系数约为0.32



# 移动窗口与动态统计函数
# 移动窗口：为了提升数据的可靠性，将某个点的取值扩大到包含这个点的区间取值，并且用区间进行判断，这个区间就是窗口
# rolling函数（数据框或序列.rolling(window = 窗口数, axis = 0/1).统计量函数(axis = 0/1)）
# 移动平均
# 使用SH_Index_new1作为分析对象，创建收盘价10日均值的序列并将其变为数据框，将10日均值收盘价与每日收盘价进行可视化
# 创建10日均值收盘价的序列
SH_Index_MA10 = SH_Index_new1["收盘价"].rolling(window = 10).mean()

# 将序列变为数据框
SH_Index_MA10 = SH_Index_MA10.to_frame()

# 修改数据框列名
SH_Index_MA10 = SH_Index_MA10.rename(columns ={"收盘价": "10日平均收盘价（MA10）"})

# 创建一个每日收盘价的数据框
SH_Index_close = SH_Index_new1["收盘价"].to_frame()

# 合并成一个包括每日收盘价、10日均值收盘价的数据框
SH_Index_new5 = pd.concat([SH_Index_close, SH_Index_MA10], axis = 1)

# 绘制图像
SH_Index_new5.plot(figsize = (9, 6), title = u"2019-2020年上证综指走势", grid = True, fontsize = 13)
# 可看出相比每日收盘价，10日均值收盘价的走势更平滑



# 移动波动率
# 以SH_Index_new1作为分析对象，创建30天时间窗口的上证综指收盘价的移动波动率并且进行可视化
# 创建30日移动波动率的序列
SH_Index_rollstd = SH_Index_new1["收盘价"].rolling(window = 30).std()

# 将序列变为数据框
SH_Index_rollstd = SH_Index_rollstd.to_frame()

# 修改数据框列名
SH_Index_rollstd = SH_Index_rollstd.rename(columns = {"收盘价": "30日收盘价的移动波动率"})

# 绘制图像
SH_Index_rollstd.plot(figsize = (9, 6), title = "2019-2020年上证综指移动波动率的走势", grid = True, fontsize = 12)
# 可看出上证综指的移动波动率本身存在较大的波动，最高波动率触及200，最低波动率仅为25



# 移动相关系数
# 变量之间的相关系数会随时间的变化而变化（尤其在金融危机期间，很多原本相关性很低的变量呈现出较高的相关性）故计算不同变量之间的移动相关系数来捕捉相关关系的变化
# 使用SH_Index_new1作为分析对象，计算60天时间窗口的上证综指开盘价、最高价、最低价以及收盘价之间的移动相关系数
# 计算移动相关系数
SH_Index_rollcorr = SH_Index_new1.rolling(window = 60).corr()

# 删除缺失值
SH_Index_rollcorr = SH_Index_rollcorr.dropna()

# 查看前5行
SH_Index_rollcorr.head()

# 查看末尾5行
SH_Index_rollcorr.tail()
# 可看出针对60天的时间窗口的移动相关系数，该数据框是从2019年4月2日开始的
