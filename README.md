<h3 align="center">
<p> 蛋白质类泛素化位点预测器</p> </h3>

该资源库包含用于保存已构建和训练的 Model1 模型的脚本以及预测器脚本。以及使用该模型构建的预测器脚本。

### Dependency

```
python                  2.7.15
codecs                  
csv                 
pickle                
Tkinter          
tkFileDialog               
tkMessageBox            
os             
pandas                  0.24.2
scipy                   1.2.3
numpy                   1.16.6
sklearn-learn           1.3.2            
```
Pse-in-One-2.0           http://bliulab.net/Pse-in-One2.0/download/ 
### Dataset

DR1_smote_ENN.mat		经过基于距离的残基特征提取和Smote-ENN重采样技术处理后的数据集1文件

100seq.txt			            预测器测试数据，100条蛋白质序列均为阳性样本

775seq.txt				    预测器测试数据，775条蛋白质序列均为阳性样本

17807seq.txt                                预测器测试数据，17807条蛋白质序列均为阴性样本


### Train and Test
#### Train
保存在数据集1上构建的模型Model1
```python
python Saved_Model.py
```


#### Test
开启蛋白质类泛素化预测器
```shell
python Predictor2_0.py
```

