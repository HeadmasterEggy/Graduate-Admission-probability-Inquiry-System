# **利用人工神经网络预测硕士录取概率**

# 一．论文介绍

## 摘要

本文研究了基于人工神经网络模型来预测学生硕士项目录取的概率。模型考虑了多种录取影响因素，包括标准化考试成绩（如GRE和TOEFL）、本科大学排名、个人陈述、推荐信、累计平均成绩点（CGPA）以及研究经历。通过分析这些因素如何影响录取结果，构建了一个能够为潜在申请者提供录取预测的工具，进而帮助他们更好地准备申请材料和选择申请策略。

 

## 引言

本文关注的指标：GRE、TOEFL成绩、大学排名、CGPA，个人陈述、推荐信和研究经历。通过训练一个人工神经网络模型来建立一个预测工具来估计申请人的录取概率。

 

 

## 人工神经网络模型介绍（ANN）

人工神经网络（Artificial Neural Networks, ANN）是一种模仿人脑处理信息的方式设计的计算模型。它由大量的节点（或称为“神经元”）组成，这些节点通过一系列的连接互相交互。神经网络的基本原理是通过模拟大脑的神经元如何相互作用来处理数据。这种模型在诸如图像和语音识别、自然语言处理和预测分析等多个领域有着广泛的应用。

 

## 网络结构

人工神经网络包括三种类型的层：

1. 输入层（Input layer）：接收输入数据，每个输入特征对应一个神经元。
2. 隐藏层（Hidden layer）：一个或多个隐藏层，每个层由多个神经元组成。这些神经元将输入层的数据通过非线性转换处理后传递给下一层。
3. 输出层（Output layer）：产生最终的输出结果，输出层的设计取决于特定的应用需求，例如分类或回归。

 

## 工作原理

\- 前向传播：数据从输入层开始，经过隐藏层，最后到输出层。每个神经元接收到来自前一层的输入，这些输入通过权重加权，并加上一个偏置项，然后通过一个激活函数来决定该神经元的输出。

\- 激活函数：是非线性函数，常见的如Sigmoid、Tanh或ReLU。激活函数的选择对网络的性能和能力有重大影响。

\- 反向传播和学习：网络通过反向传播算法进行训练。在这个过程中，网络输出的误差会被用来调整权重和偏置，通过梯度下降或其他优化技术逐渐减少误差。

## 确定问题

利用UCLA 研究生本科成绩信息数据集来训练模型并预测录取率。

 

## 数据预处理

先导入 csv 数据集，再用 panda 库info( ) 查看是否存在非空值，再 isnull( ).sum( ) 查看含有多少空值，drop( ) 去除空值。

因为异常值会对预测（特别是在回归问题中）产生巨大影响，所以我选择了管理这些异常值。所以使用了Tukey方法（Tukey JW., 1977）来检测异常值，该方法定义了一个介于分布值的第一四分位数和第三四分位数之间的四分位数间距（IQR）。如果某行数据在某个特征值上超出了（IQR ± 异常值步长）的范围，那么这一行就被视为含有异常值。

再从数值特征（GRE Score, TOEFL Score, University Rating, SOP, LOR , CGPA, Research）中检测异常值。然后将那些至少有两个异常数值的行视为异常值行。

然后去除Serial No. 列，数据处理完成。

 

## 模型训练

将数据用 MinMaxScaler进行归一化，分为20%测试集和 80% 训练集。搭建神经网络架构，测试不同隐藏层的数量对 r2 分数的影响。

 

## 模型优化

通过增加隐藏层数量和迭代 (epochs) 次数，使用 Adam 优化器训练模型自适应学习率，可以显著提高 r2分数。再通过不同模型对比，比如随机森林，决策树等，来选择最高准确率的模型。

 

## 文献学习

### **1.** 人工神经网络简介

ANNs是一种复杂的自适应系统，灵感来源于人类大脑，能够处理非线性问题并根据新数据调整其结构。这种网络特别适用于处理那些传统统计方法难以应对的动态复杂问题。论文强调了ANNs在提高疾病诊断和管理的准确性方面的潜力，同时也讨论了实施这一技术的挑战，如需要大量数据进行训练和网络架构的复杂性。随着计算技术的发展，ANNs有望在医疗领域提供更个性化和精准的解决方案，从而改变传统的医疗诊断和治疗方法。

 

### **2.** 人工神经网络中的类脑学习：综述

这篇论文探讨了人工神经网络（ANN）中融入更生物合理的学习机制的可能性与方法。尽管ANN在多个领域已取得显著成就，例如图像和语音生成、游戏以及机器人技术，它们的运作机制与生物大脑之间仍存在基本差异，特别是在学习过程方面。因此，研究者借鉴大脑的学习机制，以增强网络的学习能力和适应性。 

文章还说明了了支持大脑学习的各种机制，突触可塑性、神经调制和超可塑性等，这些通过局部和全局活动的相互作用调整神经元的连接强度，适应新信息。

论文还探讨了如何通过脉冲神经网络和反向传播衍生的局部学习算法，反馈对齐和资格传播等技术，模仿这些生物特性来提高人工系统的性能。通过这些生物启发的学习机制，不仅可以极大提升ANN的性能和适应性，还能推动神经科学和人工智能领域的进一步发展。

 

### **3.** 密集连接卷积网络

这篇论文介绍了DenseNet，一种新型的卷积神经网络结构，特点是每一层都直接连接到之前的所有层，从而有效地利用了特征重用，减少了参数数量，并改善了梯度流动，有助于解决深层网络中的梯度消失问题。这种密集连接的设计不仅提高了网络的训练效率，还在多个视觉识别任务上达到了优异的性能。

使用这样的高级神经网络结构来预测硕士录取率可以提供显著优势，主要体现在其强大的特征学习能力、减少过拟合的风险、改善梯度流动、高效的参数使用，以及适应不同数据类型的能力。DenseNet通过其密集连接的设计，可以从申请者的多维数据中提取出复杂的特征，更好地处理结构化和非结构化数据，有效提高模型在新数据上的泛化能力。这些特性是处理如硕士录取过程中的各种因素（包括成绩、经历、推荐信等）的理想选择。

 

### **4.** 用于目标检测的深度神经网络

该文介绍了一种新颖的将物体检测作为回归问题的表述方法，目的是通过DNNs预测单个图像中多个物体的位置。为了提高定位精度，该方法采用了多尺度策略，使用完整图像及其大型裁剪部分。此方法包括生成低分辨率掩模，并进行细化，以在较低的计算成本上实现像素级精度。该方法避免了需要手动设计显式捕获对象部分及其关系的模型，简化了对各种对象类别的应用。这种简单性带来了对不同类型对象（无论是刚性还是可变形）的检测性能的提高。

下面是一个最小化问题，旨在通过调整网络参数Θ来减小DNN模型输出（DNN(x; Θ)）和目标掩模m之间的差异。

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps1.jpg) 

这个公式定义了一个损失函数，用于训练深度学习模型进行图像中物体的定位。它的目标是找到一组网络参数Θ，使得网络能够尽可能准确地预测图像中物体的位置，即其边界框掩模。

 

### **5.** DNN模型的设计可能性和挑战:从终端设备的角度进行综述

论文讨论了两种量化方法：

均匀量化：将数据从浮点数转换为定点数，以减少精度和计算需求。

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps2.jpg) 

非均匀量化：这种方法特别有用，因为它考虑到权重的非均匀分布，通过诸如对数域量化和学习量化等方法改进准确度。

 

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps3.jpg) 

这些量化公式的目的是减小由于量化而造成的信息损失，同时减少模型参数的存储和计算需求。通过这种方式，即使是复杂的深度神经网络模型也可以被有效地部署在资源有限的设备上，而不会显著牺牲性能。

 

[1] Introduction to artificial neural networks, January 2008European Journal of Gastroenterology & Hepatology 19(12):1046-54, DOI:10.1097/MEG.0b013e3282f198a0, SourcePub: Med

[2] M. S. B. M. M. P. W. (2014). Research Paper on Basic of Artificial Neural Network. **International Journal on Recent and Innovation Trends in Computing and Communication**, **2**(1), 96–100. https://doi.org/10.17762/ijritcc.v2i1.2920

[3] Densely Connected Convolutional Networks, arXiv:1608.06993 [cs.CV]. https://arxiv.org/abs/1608.06993

[4] Deep Neural Networks for Object Detection, https://papers.nips.cc/paper_files/paper/2013/hash/f7cade80b7cc92b991cf4d2806d6bd78-Abstract.html

[5] Design possibilities and challenges of DNN models: a review on the perspective of end devices, Artifcial Intelligence Review, https://doi.org/10.1007/s10462-022-10138-z

 

 

# **二．**选择神经网络原因

录取率受多个因素影响，其中包括考试成绩、个人陈述、推荐信等，而ANN能够有效地捕捉并建模这些因素之间复杂的非线性关系，提高了预测的准确性。其次，ANN能够整合不同来源的数据，进行综合分析，从而提高了预测的全面性和准确性。此外，ANN具有很高的灵活性和可调性，能够根据具体情况进行调整和优化，适应不同学校、不同专业的录取要求。最重要的是，ANN是一种数据驱动的方法，能够从大量的历史数据中学习模式和规律，基于学习到的知识进行预测，这使得预测更加准确和可靠。

 

# **三．**实现过程

## **1.** 数据集准备

Admission_Predict.csv （共 10w 条数据）	

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps4.jpg) 

数据有 8 个特征，分别是GRE、TOEFL成绩、大学排名、CGPA，个人陈述、推荐信和研究经历。

 

## **2.** 数据预处理

导入数据

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps5.jpg) 

查看数据信息

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps6.jpg) 

查看数据是否有空值

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps7.jpg) 

使用Tukey方法来检测异常值，如果某行数据在某个特征值上超出了（IQR ± 异常值步长）的范围，那么这一行就被视为含有异常值。

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps8.jpg) 

显示异常值行

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps9.jpg) 

不存在异常值，因为所有的值都在一个固定的范围内，没有一个值会低于或超过这个范围，因此不产生异常值。

 

去除 Serial No. 列

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps10.jpg) 

创建热图

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps11.jpg) 

 

可以看到，录取的机会与CGPA高度相关，GRE和托福成绩也是相关的。

从上面的配对图推断:

GRE成绩、托福成绩、CGPA成绩均呈线性相关关系 无论如何，研究型学生往往得分更高。

 

循环绘制每个列的分布图

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps12.jpg) 

 

划分数据集

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps13.jpg) 

特征缩放归一化

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps14.jpg) 

 

## **3.** 模型构建

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps15.jpg) 

采用人工神经网络架构，7个神经元的密集连接层，Relu作为激活函数，1 个神经元的输出层，使用线性激活函数

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps16.jpg) 

训练完毕输出 r2 得分

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps17.jpg) 

R2 得分呈负数表示回归模型的性能比拟合数据的水平线差。这表明该模型无法捕获特征和目标变量之间的任何有意义的关系，从而导致预测性能较差。

 

损失曲线对比

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps18.jpg) 

## **4.** 模型优化

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps19.jpg) 

15个神经元的全连接层，3 个隐藏层，神经元的数量增加，选用 Adam优化器自适应学习率，迭代次数增加到 100。

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps20.jpg) 

r2得分得到显著提升

 

## **5.** 测试数据

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps21.jpg) 

预测前十个数据

 

## **6.** 其他模型对比

使用决策树，随机森林，knn，线性回归，AdaBoost 模型对比

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps22.jpg) 

结果显示线性回归 r2 得分最好，说明准确率最高。

 

用线性回归预测前十个数据。

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps23.jpg) 

保存模型

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps24.jpg) 

 

## **7.** 界面设计

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps25.jpg) 

输入信息选择不同模型得出预测结果，下面生成整体趋势图

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps26.jpg) 

 

后端查看选择信息

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps27.jpg) 

 

线性回归

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps28.jpg) 

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps29.jpg) 

 

## **8.** 具体代码

预测代码

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps30.jpg) 

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps31.jpg) 

前端调用

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps32.jpg) 

 

生成图片代码

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps33.jpg) 

前端调用

![img](file:////Users/joey/Library/Containers/com.kingsoft.wpsoffice.mac/Data/tmp/wps-joey/ksohtml//wps34.jpg) 