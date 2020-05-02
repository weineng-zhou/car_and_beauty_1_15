# car_and_beauty_1_15
汽车美女图像识别分类

项目名称：car_and_beauty_1_15
项目描述：通过训练汽车和美女图片数据得到图像识别模型，
	  然后对未知标签的美女或汽车图片进行识别分类

这里用的tensorflow是1.15.2版本，高版本暂不支持，2.2以上版本可以参考tensorflow官网实例

安装依赖模块
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 

data 文件夹分为cars和beauty用来训练和测试模型。

pred文件夹可以随意放8张美女或汽车的图片，让模型进行识别分类。

代码最后可以更改预测图片个数，随意测多少张，最好是偶数
