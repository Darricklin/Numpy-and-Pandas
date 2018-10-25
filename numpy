import numpy as np
np.__version__
'1.14.0'
什么是numpy？

Numpy即Numeric python ，python经过扩展以后，可以支持数组和矩阵，并且包含大量的函数可以用于计算矩阵和数组

【注】numpy是数据分析的和机器学习中的基本工具，后面的许多的框架和工具都是基于numpy，如：pandas、scipy、matplotlib、sklearn、TensorFlow
一、创建numpy numpy中最基础数据结构就是ndarray：即数组


1. 使用np.array()由python list创建

l = [1,2,3,4,3.21352,True,"asdfdassfda"]
nd=np.array(l)
nd
array(['1', '2', '3', '4', '3.21352', 'True', 'asdfdassfda'], dtype='<U32')
注意：1、数组的元素类型必须相同 2、如果通过列表来创建数组，列表中的元素的类型如果不一样，就会被变成同一数据类型（str>float>int>bealean） 【注意】客观世界中的信息都可以被转化成数组，比如声音、图片等


import matplotlib.pyplot as plt# 导入数据可视化框架
#%matplotlib inline # 加入一个魔法指令，在调用plt的那些cell结束的地方，自动的补上一个plt.show()，目前版本不需要这行代码
UsageError: unrecognized arguments: # 加入一个魔法指令，在调用plt的那些cell结束的地方，自动的补上一个plt.show()，目前版本不需要这行代码

girl=plt.imread('./0A-波波老师(001C42B29C13)/day01  numpy，jupter,ipython/2-numpy/source/girl.jpg')
plt.imshow(girl)
<matplotlib.image.AxesImage at 0x9272240>


girl.shape
(900, 1440, 3)

girl.dtype
dtype('uint8')

plt.imshow(girl,cmap='gray')
<matplotlib.image.AxesImage at 0x91ee978>


g=girl[0:400,600:1000]
plt.imshow(g，cmap='gray')#cmap='gray'表示灰度显示，但是对于三维数组不起作用
<matplotlib.image.AxesImage at 0x923f2e8>


l=[[0.5,0.8,0.9],[0.1,0.5,0.3],[0.8,0.3,0.7]]
nd1=np.array(l)
plt.imshow(nd1,cmap='gray')#cmap='gray'表示灰度显示，对二维数组起作用
<matplotlib.image.AxesImage at 0xa1a99b0>

使用np的routines函数创建
(1)np.ones(shape,dtype=None,order='C')


np.ones(shape=(2,3,4,5,6))
# 一个必要参数shape代表数组的形状，要求传递一个列表（或者元组）,参数中数字的个数代表创建出来的数组维度，每个数字值代表该维度上有多少个单元
#简写成  np.ones((2,3,4,5,6))

boy=np.ones(shape=(168,200,3))
plt.imshow(boy,cmap='gray')#因为里面的数值都是用1填充的，所以显示为白色
(2) np.zeros(shape,dtype="float",order="C")


boy1=np.zeros((150,300,3))
plt.imshow(boy1,cmap='gray')#因为里面的数值都是用1填充的，所以显示为白色
(3) np.full(shape,fill_value,dtype=None)#fill_value表示填充值


boy2=np.full((2,3),fill_value=10)
plt.imshow(boy2)
（4）np.eye(N,M,k=0,dtype='float')#返回一个对角线为1，其余为零的矩阵，如果k改变，则对角线移动相应的索引值k


np.eye(4,5,k=0)
array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.]])

np.eye(4,5,k=1)
array([[0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
(5）np.linspace(start,stop,num=50)


np.linspace(0,10,11)# 从start到stop平均等分成num份
array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])

np.logspace(0,10,11)
# 对数切分把一个以10为底，以10的start次方到10的stop次方，按指数切分成num份
array([1.e+00, 1.e+01, 1.e+02, 1.e+03, 1.e+04, 1.e+05, 1.e+06, 1.e+07,
       1.e+08, 1.e+09, 1.e+10])
(6）np.arange([start,]stop,[step,]dtype=None) "[]"中是可选项


np.arange(10,)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

np.arange(2,10,3)
array([2, 5, 8])
(7）np.random.randint(low,high=None,size=None,dtype='I')#low表示在该值以上，high表示在该值以下，单写一个值表示最大为该值


np.random.randint(low=100,high=200,size=(100,100,3))#不能读取成图片，dtype不符合要求
dtype('int32')
（8）np.random.randn(d0,d1,...,dn) 从第一维度到第n维度生成一个数组，数组中的数字符合标准正态分布


np.random.randn(3,4,3)
(9）np.random.normal(loc=0.0,scale=1.0,size=None)#loc表示平均值，scale表示标准差，size表示输出的大小


boy3=np.random.normal(loc=180,scale=20,size=(100,200,3))
plt.imshow(boy3)
<matplotlib.image.AxesImage at 0xc1ec208>

（10）np.random.random(size=None)


img1=np.random.random((100,150,3))
plt.imshow(img1)
<matplotlib.image.AxesImage at 0xc4ac048>

二、ndarray的属性 数组的常用属性：

维度 ndim， 大小 size， 形状 shape， 元素类型 dtype， 每项大小 itemsize， 数据 data


img1.ndim
3

img1.size
45000

img1.shape
(100, 150, 3)

img1.dtype
dtype('float64')
三、ndarray的基本操作
1、索引
nd=np.random.randint(10,size=5)
nd[2]
6

l = [[1,2,3],
     [4,5,6],
     [7,8]
    ]
l
[[1, 2, 3], [4, 5, 6], [7, 8]]

l[0][1]
2

np.array(l)# 【注意】数组中所有的维度对应的数值的个数必须一样，数组必须要求m*n的形式
array([list([1, 2, 3]), list([4, 5, 6]), list([7, 8])], dtype=object)

nd = np.random.randint(0,10,size=(3,4))
nd
array([[8, 1, 5, 1],
       [0, 7, 9, 8],
       [7, 2, 1, 0]])

nd[0]
array([8, 1, 5, 1])

nd[2][1] # 分多次查找
2
列表和数组的索引区别


nd[2,1] # 如果[]中写多个索引用“，”隔开，就是要查找多个维度
2

l[2,2] # 列表不能这样查找
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-103-b1f48e1808da> in <module>()
----> 1 l[2,2] # 列表不能这样查找

TypeError: list indices must be integers or slices, not tuple


nd[[1,2,1,0,2,1]] # 查找多个元素，[]中传一个列表，得到的结果会按照列表中下标的次序来排列
array([[0, 7, 9, 8],
       [7, 2, 1, 0],
       [0, 7, 9, 8],
       [8, 1, 5, 1],
       [7, 2, 1, 0],
       [0, 7, 9, 8]])

l[[1,1,2]] # 列表也不能这样写

nd
array([[8, 1, 5, 1],
       [0, 7, 9, 8],
       [7, 2, 1, 0]])

nd[[1,2,1,0],[1,2,1,0]] # 如果用两个列表作为索引，要求两个列表的元素个数必须一致，两个列表的元素对应，在进行遍历
array([7, 1, 7, 8])
2、切片 列表的切片无法取到元素中的值（只能切行），数组可以多维度切片（即可切行也可切列）


l = [[1,2,3],
     [4,5,6],
     [7,8]
    ]
l
[[1, 2, 3], [4, 5, 6], [7, 8]]

l[:2]
l[1:][:1]
# l[:1,:2]
[[4, 5, 6]]

nd1=np.random.randint(100,size=(3,4))
nd1
array([[79,  0, 88, 20],
       [38, 81, 97, 76],
       [51, 98, 60, 24]])

nd1[0:2]
array([[79,  0, 88, 20],
       [38, 81, 97, 76]])

nd1[0:2,:2]# 对于数组可以对多个维度进行切片
array([[79,  0],
       [38, 81]])

nd1[::2,::2]# 对于数组可以对多个维度进行切片,step为2
array([[79, 88],
       [51, 60]])

nd1[2::-2] # 如果步长为负，代表从起点开始后前面数
array([[51, 98, 60, 24],
       [79,  0, 88, 20]])

nd1[2::-2,3::-2]# 如果步长为负，代表从起点开始后前面数
array([[24, 98],
       [20,  0]])

plt.imshow(girl[::-1,::-1,::-1])#图片上下颠倒，左右颠倒，颜色颠倒
<matplotlib.image.AxesImage at 0xe324940>


plt.imshow(girl[::2,::2])#缩小图片的size
<matplotlib.image.AxesImage at 0xe747358>


girl2=plt.imread('./0A-波波老师(001C42B29C13)/day01  numpy，jupter,ipython/2-numpy/source/meinv.jpg')
girl3=girl2[::2,::2]
plt.imshow(girl3)
<matplotlib.image.AxesImage at 0xed6e4e0>


tigger=plt.imread('./0A-波波老师(001C42B29C13)/day01  numpy，jupter,ipython/2-numpy/source/tigger.jpg')
tigger.flags.writeable=True   #文件是只读模式，改为可写
plt.imshow(tigger)
<matplotlib.image.AxesImage at 0xedcc2b0>


tigger[0:492,250:600]=girl3
plt.imshow(tigger)
<matplotlib.image.AxesImage at 0xee305c0>

3、变形

reshape() # reshape函数只是把传入的nd数组根据指定的形状重新排列，并且把排列以后的结果返回出去，原数组大小不变

resize() # resize函数把原数组安装指定的形状重新排列，这个改变发生在原数组本身上


nd = np.random.randint(0,100,size=(3,4))
nd
array([[85, 67, 61, 69],
       [62, 79, 55, 83],
       [11, 15, 37, 19]])

nd.reshape((2,1,2,1)) # 变形前后size必须一样

nd.reshape((2,2,3)) 
# reshape函数只是把传入的nd数组根据指定的形状重新排列，并且把排列以后的结果返回出去，原数组大小不变
array([[[85, 67, 61],
        [69, 62, 79]],

       [[55, 83, 11],
        [15, 37, 19]]])

nd
array([[[[85, 67, 61],
         [69, 62, 79]],

        [[55, 83, 11],
         [15, 37, 19]]]])

nd.resize((1,2,2,3)) 
nd
# resize函数把原数组安装指定的形状重新排列，这个改变发生在原数组本身上 
array([[[[85, 67, 61],
         [69, 62, 79]],

        [[55, 83, 11],
         [15, 37, 19]]]])
4、级联 级联：按照指定维度把多个数组链接在一起形成一个新数组的过程

关于级联需要注意的问题：

1）参与级联的数组维度必须一致
2）形状必须一致（axis指定的那个维度抛开以后，剩余的形状如果一致才能级联）
3）级联的方向有axis来控制，每次只能指定一个方向

nd1 = np.random.randint(0,100,size=(4,4))
nd2 = np.random.randint(0,100,size=(3,4))
print(nd1)
print(nd2)
[[88 17 20 62]
 [26 23 51 14]
 [84 40 57 12]
 [29 46 34 80]]
[[17 50 14 48]
 [32 14 62 77]
 [95 32  9 59]]

# 将nd1和nd2进行级联
np.concatenate([nd1,nd2],axis=0)
# 参数1，是一个列表，参与级联的数组放在这个列表中
# 参数2，axis代表级联的方向（默认是0代表在第0个维度上级联）
array([[88, 17, 20, 62],
       [26, 23, 51, 14],
       [84, 40, 57, 12],
       [29, 46, 34, 80],
       [17, 50, 14, 48],
       [32, 14, 62, 77],
       [95, 32,  9, 59]])

np.concatenate([nd1,nd2],axis=1) # 如果对列进行级联，要求行数必须一致；同理如果进行行级联要求列数必须一致

nd3 = np.random.randint(0,100,size=(4,3))
nd3
array([[31, 16, 38],
       [75, 55, 75],
       [ 9, 41, 73],
       [95,  7, 54]])

np.concatenate([nd1,nd3],axis=1)
array([[88, 17, 20, 62, 31, 16, 38],
       [26, 23, 51, 14, 75, 55, 75],
       [84, 40, 57, 12,  9, 41, 73],
       [29, 46, 34, 80, 95,  7, 54]])

nd4 = np.random.randint(0,100,size=(1,2,3))
nd5 = np.random.randint(0,100,size=(1,4,3))
print(nd4)
print(nd5)
[[[64  9 13]
  [ 0  1  0]]]
[[[29 74 22]
  [ 4 10 13]
  [65 23 58]
  [ 5 71  9]]]
（1)形状一致才能级联


np.concatenate([nd4,nd5],axis=1)
# axis指定的那个维度抛开以后，剩下形状如果一致，就可以级联，否则不能级联
array([[[64,  9, 13],
        [ 0,  1,  0],
        [29, 74, 22],
        [ 4, 10, 13],
        [65, 23, 58],
        [ 5, 71,  9]]])

np.concatenate([nd4,nd5],axis=[1,2]) # 级联的方向只能有一个
（2）维度一致才能级联


nd6 = np.random.randint(0,100,size=4)
nd6
array([45, 29, 48, 47])

np.concatenate([nd1,nd6],axis=1)
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-161-0fe4c85adf9f> in <module>()
----> 1 np.concatenate([nd1,nd6],axis=1)

ValueError: all the input arrays must have same number of dimensions

对于二维数组还有两个方法需要掌握
hstack 把二维数组变成一维（两列变成一行） vstack 把一维数组变成二维（一行变成一列）


nd = np.random.randint(0,10,size=(10,2))
nd
array([[7, 7],
       [9, 0],
       [4, 8],
       [9, 5],
       [3, 9],
       [2, 6],
       [1, 5],
       [9, 7],
       [2, 5],
       [8, 2]])

nd3 = np.hstack(nd)
nd3
# 这个函数的作用：把原来按列排（二维数组是按列来排）的变成按行排（只有一维数组是水平排列）
array([7, 7, 9, 0, 4, 8, 9, 5, 3, 9, 2, 6, 1, 5, 9, 7, 2, 5, 8, 2])

np.vstack(nd) # 对二维数组用vstack无效
array([[7, 7],
       [9, 0],
       [4, 8],
       [9, 5],
       [3, 9],
       [2, 6],
       [1, 5],
       [9, 7],
       [2, 5],
       [8, 2]])

nd2 = np.random.randint(0,10,size=10)
nd2
array([2, 6, 4, 4, 8, 5, 3, 0, 5, 1])

np.vstack(nd2) # 把原来按行排列的数组变成按列排序
array([[2],
       [6],
       [4],
       [4],
       [8],
       [5],
       [3],
       [0],
       [5],
       [1]])

np.vstack(nd3)
5、切分 切分就是把一个数组切成若干个

hsplit() 沿y轴方向切分（纵向）

vsplit() 沿x轴方向切分（横向）

split() 按照维度切分


nd = np.random.randint(0,100,size=(5,6))
nd
array([[18, 71,  4, 65, 21, 66],
       [31, 74, 95, 24, 59, 22],
       [71, 39, 25, 61, 50, 78],
       [32, 94, 88, 89, 53, 25],
       [14, 58,  7, 14, 25, 65]])

# 沿y轴方向切分（纵向）
np.hsplit(nd,[1,3,5,8,100])
# 参数1，代表被切分的数组，参数2，代表切分的位置
[array([[18],
        [31],
        [71],
        [32],
        [14]]), array([[71,  4],
        [74, 95],
        [39, 25],
        [94, 88],
        [58,  7]]), array([[65, 21],
        [24, 59],
        [61, 50],
        [89, 53],
        [14, 25]]), array([[66],
        [22],
        [78],
        [25],
        [65]]), array([], shape=(5, 0), dtype=int32), array([], shape=(5, 0), dtype=int32)]

np.vsplit(nd,[1,3,4])#沿x轴方向切分（横向）
[array([[18, 71,  4, 65, 21, 66]]), array([[31, 74, 95, 24, 59, 22],
        [71, 39, 25, 61, 50, 78]]), array([[32, 94, 88, 89, 53, 25]]), array([[14, 58,  7, 14, 25, 65]])]

np.split(nd,[1,2],axis=1)
[array([[18],
        [31],
        [71],
        [32],
        [14]]), array([[71],
        [74],
        [39],
        [94],
        [58]]), array([[ 4, 65, 21, 66],
        [95, 24, 59, 22],
        [25, 61, 50, 78],
        [88, 89, 53, 25],
        [ 7, 14, 25, 65]])]

nd1 = np.random.randint(0,100,size=(3,4,5))
nd1
array([[[28,  3, 75, 65, 89],
        [74, 18, 83, 62, 48],
        [21,  4, 16,  8, 69],
        [ 9, 97,  0,  7, 13]],

       [[93, 44, 15, 87, 67],
        [99, 72, 95, 49, 86],
        [80, 38, 70, 74, 18],
        [ 3, 21, 21, 75, 86]],

       [[56, 72, 58, 67, 19],
        [87,  5, 61, 83, 23],
        [61, 63, 94, 94, 42],
        [79,  7, 96, 23, 96]]])

np.split(nd1,[1,2],axis=2)
[array([[[28],
         [74],
         [21],
         [ 9]],
 
        [[93],
         [99],
         [80],
         [ 3]],
 
        [[56],
         [87],
         [61],
         [79]]]), array([[[ 3],
         [18],
         [ 4],
         [97]],
 
        [[44],
         [72],
         [38],
         [21]],
 
        [[72],
         [ 5],
         [63],
         [ 7]]]), array([[[75, 65, 89],
         [83, 62, 48],
         [16,  8, 69],
         [ 0,  7, 13]],
 
        [[15, 87, 67],
         [95, 49, 86],
         [70, 74, 18],
         [21, 75, 86]],
 
        [[58, 67, 19],
         [61, 83, 23],
         [94, 94, 42],
         [96, 23, 96]]])]
6、副本


nd = np.random.randint(0,10,size=10)
nd
array([5, 8, 8, 5, 3, 1, 8, 0, 1, 1])

nd1 = nd # 引用拷贝，只拷贝地址，不拷贝数组本身

nd[1] = 1000

nd1
array([   5, 1000,    8,    5,    3,    1,    8,    0,    1,    1])

nd2 = nd.copy() # 副本考本，给nd引用的那个数组拷贝了一个副本

nd2[3] = 1200

nd
array([   5, 1000,    8,    5,    3,    1,    8,    0,    1,    1])

nd2
array([   5, 1000,    8, 1200,    3,    1,    8,    0,    1,    1])
用列表创建数组，有没有副本拷贝？ 从列表到数组进行了副本的拷贝


l = [1,2,3,4,5]

nd = np.array(l)
nd
array([1, 2, 3, 4, 5])

l[0] =10000

nd
array([1, 2, 3, 4, 5])
四、ndarray的聚合操作 聚合操作指的就是对数组内部的数据进行的某些操作

1.求和
聚合的规律：（1）通过axis来制定聚合的方向 （2）axis=x则第x个维度就会消失，取而代之的是对这个维度上聚合的结果


nd=np.random.randint(0,10,size=(3,4))
nd
array([[8, 9, 5, 5],
       [7, 4, 7, 7],
       [7, 3, 0, 7]])

nd.sum()#完全聚合，把整个数组中的所有元素全部聚合在一起
69

nd.sum(axis=0)#对第0个维度进行聚合
array([22, 16, 12, 19])

nd.sum(axis=1)#对第1个维度进行聚合
array([27, 25, 17])
推广：多维


nd=np.random.randint(10,size=(2,3,4))
nd
array([[[0, 9, 2, 5],
        [6, 0, 4, 5],
        [4, 8, 0, 6]],

       [[8, 4, 8, 0],
        [1, 8, 3, 3],
        [7, 3, 3, 6]]])

nd.sum()
103

nd.sum(axis=0)
array([[ 8, 13, 10,  5],
       [ 7,  8,  7,  8],
       [11, 11,  3, 12]])

nd.sum(axis=1)
array([[10, 17,  6, 16],
       [16, 15, 14,  9]])

nd.sum(axis=2)
array([[16, 15, 18],
       [20, 15, 19]])
练习：给定一个4维矩阵，如何得到最后两维的和？


nd=np.random.randint(10,size=(2,3,4,5))
nd
写法一


nd1=nd.sum(axis=3)
nd1

nd2=nd1.sum(axis=2)
nd2
array([[ 82,  99, 102],
       [ 94,  86,  84]])
写法二


nd.sum(axis=2).sum(axis=2)
array([[ 82,  99, 102],
       [ 94,  86,  84]])

nd.sum(axis=3).sum(axis=2)
array([[ 82,  99, 102],
       [ 94,  86,  84]])

nd.sum(axis=-1).sum(axis=-1)
array([[ 82,  99, 102],
       [ 94,  86,  84]])
写法三


nd.sum(axis=(-1,-2))
array([[ 82,  99, 102],
       [ 94,  86,  84]])
2.求最值


nd
array([[[[3, 8, 6, 0, 2],
         [4, 5, 3, 5, 9],
         [2, 9, 1, 2, 5],
         [7, 7, 0, 2, 2]],

        [[4, 2, 8, 7, 6],
         [8, 9, 4, 3, 1],
         [4, 8, 7, 9, 8],
         [8, 1, 1, 0, 1]],

        [[0, 8, 4, 5, 9],
         [7, 6, 8, 3, 4],
         [5, 1, 5, 9, 5],
         [4, 4, 1, 8, 6]]],


       [[[9, 9, 9, 8, 8],
         [3, 6, 0, 4, 0],
         [3, 1, 8, 7, 5],
         [2, 8, 1, 0, 3]],

        [[9, 9, 2, 2, 2],
         [2, 0, 3, 0, 3],
         [2, 6, 4, 3, 9],
         [6, 6, 5, 4, 9]],

        [[9, 6, 6, 3, 2],
         [7, 2, 0, 6, 3],
         [4, 4, 4, 9, 0],
         [8, 4, 0, 5, 2]]]])

nd.max()
9

nd.max(axis=0)
array([[[9, 9, 9, 8, 8],
        [4, 6, 3, 5, 9],
        [3, 9, 8, 7, 5],
        [7, 8, 1, 2, 3]],

       [[9, 9, 8, 7, 6],
        [8, 9, 4, 3, 3],
        [4, 8, 7, 9, 9],
        [8, 6, 5, 4, 9]],

       [[9, 8, 6, 5, 9],
        [7, 6, 8, 6, 4],
        [5, 4, 5, 9, 5],
        [8, 4, 1, 8, 6]]])

nd.max(axis=1)
array([[[4, 8, 8, 7, 9],
        [8, 9, 8, 5, 9],
        [5, 9, 7, 9, 8],
        [8, 7, 1, 8, 6]],

       [[9, 9, 9, 8, 8],
        [7, 6, 3, 6, 3],
        [4, 6, 8, 9, 9],
        [8, 8, 5, 5, 9]]])
3、其他聚合操作


 Function Name  NaN-safe Version    Description
    np.sum  np.nansum   Compute sum of elements
    np.prod np.nanprod  Compute product of elements
    np.mean np.nanmean  Compute mean of elements
    np.std  np.nanstd   Compute standard deviation
    np.var  np.nanvar   Compute variance
    np.min  np.nanmin   Find minimum value
    np.max  np.nanmax   Find maximum value
    np.argmin   np.nanargmin    Find index of minimum value
    np.argmax   np.nanargmax    Find index of maximum value
    np.median   np.nanmedian    Compute median of elements
    np.percentile   np.nanpercentile    Compute rank-based statistics of elements
    np.any  N/A Evaluate whether any elements are true
    np.all  N/A Evaluate whether all elements are true
    np.power 幂运算

np.nan在numpy代表什么都没有，是一个float类型，可以和任何类型的数字进行数学运算，结果任然
为nan

nd2=np.array([12,22,34,23,np.nan])
nd2.sum()
nan

np.nansum(nd2)
91.0

np.nanmean(nd2)
22.75

对于一些带有nan数组在聚合的时候再用带有nan前缀的聚合方法来聚合，此时会将带nan的那些项剔除

nd=np.random.randint(10,size=(5,5))
nd
array([[3, 4, 7, 3, 0],
       [7, 4, 0, 9, 8],
       [3, 2, 2, 9, 6],
       [6, 2, 6, 8, 1],
       [2, 7, 6, 6, 1]])

矩阵的排序

np.sort(nd)
array([[0, 3, 3, 4, 7],
       [0, 4, 7, 8, 9],
       [2, 2, 3, 6, 9],
       [1, 2, 6, 6, 8],
       [1, 2, 6, 6, 7]])

nd[:,3]#取第三列
array([4, 8, 6, 6, 6])

ind=np.argsort(nd[:,3])
ind
array([0, 2, 3, 4, 1], dtype=int64)

nd[ind]
array([[0, 3, 3, 4, 7],
       [2, 2, 3, 6, 9],
       [1, 2, 6, 6, 8],
       [1, 2, 6, 6, 7],
       [0, 4, 7, 8, 9]])
五、ndarray的矩阵操作

1、基本矩阵操作（即加减乘除）


nd=np.random.randint(10,size=(3,4))
nd
array([[5, 9, 4, 3],
       [5, 3, 0, 2],
       [3, 4, 7, 5]])

nd1=np.random.randint(10,size=(3,3))
nd1
array([[3, 1, 9],
       [7, 7, 6],
       [1, 8, 9]])

nd+nd1#形状不同的矩阵不能相加减
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-51-092b64f00d97> in <module>()
----> 1 nd+nd1

ValueError: operands could not be broadcast together with shapes (3,4) (3,3) 

乘除（这里的乘除指的是矩阵和常数之间的乘除）


nd*8
array([[40, 72, 32, 24],
       [40, 24,  0, 16],
       [24, 32, 56, 40]])

10/nd
C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: RuntimeWarning: divide by zero encountered in true_divide
  """Entry point for launching an IPython kernel.
array([[2.        , 1.11111111, 2.5       , 3.33333333],
       [2.        , 3.33333333,        inf, 5.        ],
       [3.33333333, 2.5       , 1.42857143, 2.        ]])

nd/10
array([[0.5, 0.9, 0.4, 0.3],
       [0.5, 0.3, 0. , 0.2],
       [0.3, 0.4, 0.7, 0.5]])
矩阵积 两个矩阵的乘必须满足第一个矩阵的列数等于第二个矩阵的行数相等，否则无法相乘。第一个矩阵的行数决定结果的行数，第二个矩阵的列数决定了结果的列数。


np.dot(nd,nd1)#不是随便的两个矩阵就可以相乘
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-57-048c1d7ca112> in <module>()
----> 1 np.dot(nd,nd1)#不是随便的两个矩阵就可以相乘

ValueError: shapes (3,4) and (2,3) not aligned: 4 (dim 1) != 2 (dim 0)


nd1=np.random.randint(10,size=(2,3))
nd2=np.random.randint(10,size=(3,4))
print(nd1)
print(nd2)
[[6 7 5]
 [4 4 8]]
[[2 0 5 6]
 [9 4 0 9]
 [4 8 0 9]]

np.dot(nd1,nd2)
array([[ 95,  68,  30, 144],
       [ 76,  80,  20, 132]])

nd3=np.random.randint(10,size=(3,4))
nd4=np.random.randint(10,size=(4,5))
print(nd3)
print(nd4)
[[2 7 3 4]
 [4 2 3 4]
 [3 5 5 5]]
[[5 3 8 8 9]
 [8 4 8 7 0]
 [8 1 3 2 9]
 [4 4 3 9 3]]

np.dot(nd3,nd4)
array([[106,  53,  93, 107,  57],
       [ 76,  39,  69,  88,  75],
       [115,  54,  94, 114,  87]])

2、广播机制
ndarray的广播机制的两条规则
        （1）为缺失维度补1
        （2）假定缺失的元素已有值填充
广播机制的原理：
        （1）两个矩阵相加减必须满足形状相同。否则无法相加减
        （2）numpy里面提供了一种广播机制，可以实现矩阵与向量，与常数之间的运算
        （3）矩阵和行向量（一位数组）之间运算，满足行向量中元素个数必须和矩阵的列数一致，然后拿行向量已有的行补全成和矩阵形状一样的一个矩阵，再相加减。
        （4）矩阵和列向量（即n行1列的数组）之间相加减，要满足向量的行数和矩阵的行数一致，然后向量已有的列补全成和矩阵形状一样的矩阵，再相加减。
        （5）矩阵和常数之间的相加减，直接拿常数补全成矩阵的形状，然后相加减

nd=np.random.randint(10,size=5)#行向量
nd
array([6, 0, 0, 2, 5])

nd1=np.random.randint(10,size=(5,1))#列向量
nd1
array([[6],
       [8],
       [0],
       [1],
       [0]])

nd2=np.random.randint(10,size=(5,5))

nd2+nd1
array([[13, 11,  6, 14,  9],
       [10, 16, 16,  8,  9],
       [ 2,  0,  3,  6,  2],
       [ 7,  2,  6,  9,  7],
       [ 6,  8,  0,  3,  9]])

nd2+nd
array([[13,  5,  0, 10,  8],
       [ 8,  8,  8,  2,  6],
       [ 8,  0,  3,  8,  7],
       [12,  1,  5, 10, 11],
       [12,  8,  0,  5, 14]])

nd=np.random.randint(10,size=(3,3))
nd1=np.random.randint(10,size=3)
nd2=np.random.randint(10,size=(3,1))
print(nd)
print(nd1)
print(nd2)
[[5 1 3]
 [6 3 1]
 [6 6 9]]
[9 3 7]
[[2]
 [6]
 [7]]

nd+nd1
array([[14,  4, 10],
       [15,  6,  8],
       [15,  9, 16]])

nd+nd2
array([[ 7,  3,  5],
       [12,  9,  7],
       [13, 13, 16]])

​

​
