# C2DFNet
This is the official implementaion of TMM paper "C2DFNet: Criss-Cross Dynamic Filter Network for RGB-D Salient Object Detection".

Miao Zhang, Shunyu Yao, Beiqi Hu, [Yongri Piao](http://ice.dlut.edu.cn/yrpiao/), Wei Ji.

## Prerequisites
+ Ubuntu 16
+ PyTorch 1.10.0
+ CUDA 11.3
+ Python 3.8

## Training and Testing Datasets
Training dataset
* [Download Link](https://pan.baidu.com/s/14cGEwcCRulWDOuKNIjuGCg). Code: 0fj8

Testing dataset
* [Download Link](https://pan.baidu.com/s/1Yp5YtVIBB3-9PMFruYhxSw). Code: f7vk


Download pretrained model from [here](). Code: 
* Modify your path of testing dataset in test_depth
* Run test.py to inference saliency maps
* Saliency maps generated from the RGB stream can be downnloaded from [here](). Code: 

```shell
python test.py
```


## Contact and Questions
Contact: Zhengkun Rong. Email: yao_shunyu@foxmail.com or ysyfever-few@mail.dlut.edu.cn
