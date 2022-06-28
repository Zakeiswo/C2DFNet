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

## Testing
Download pretrained model from [here](https://pan.baidu.com/s/1_3rA5Y_jtUXzIJO8imZz2g). Code: qcra
* Modify your path of testing dataset in test.py
* Run test.py to inference saliency maps
* Saliency maps generated from the model can be downnloaded from [here](https://pan.baidu.com/s/10UQOmUbDWDvw87gGAjeM-A). Code: hp32

```shell
python test.py
```


## Contact and Questions
Contact: Shunyu Yao. Email: yao_shunyu@foxmail.com or ysyfeverfew@mail.dlut.edu.cn
