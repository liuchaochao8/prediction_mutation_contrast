#导入必须的包
import matplotlib.pyplot as plt
import numpy as np
#-----------  打开txt文件   ----------
file = open('loss.txt')
#-----------  逐行读取文件内的数据  ------------
data = file.readlines()
#-----------  根据自己的需要查看data的内容  ---------
#print(data)
'''
txt文件的数值为y轴的数据
所以x要根据y的个数有序生成
'''
#------ x轴数据有序生成150个（根据自己的横坐标范围自己修改范围）  ----
x = np.arange(0,100)
#----------  新建一个空的列表，用于存储上一步逐行读取的data  ------------
y = []
#---------- 用循环的方式添加进列表  -----------
for num in data:
	#------split用于将每一行数据用逗号分割成多个对象-----
    #------取分割后的第0列，转换成float格式后添加到列表中-------
    y.append(float(num.split(',')[0]))
#---------------    输出图    ----------------------
#---------   可以理解为在图上加载x和y的数据   label为关于x和y曲线的标签------------
pic = plt.plot(x,y,label='loss')
#---------   x轴的小标题   -------------
plt.xlabel('Epoch')
#---------   y轴的小标题   -------------
plt.ylabel('Loss')
#---------   整个图的标题  ----------
plt.title('loss')
plt.legend()
plt.show()
