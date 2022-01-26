# Vit-ImageClassification

## Introduction
This project uses ViT to perform image classification tasks on DATA set CIFAR10. The implement of Vit and pretrained weight are from https://github.com/asyml/vision-transformer-pytorch. 

![The architecture of ViT](https://github.com/Kaicheng-Yang0828/Vit-ImageClassification/blob/main/pic/VIT.png)

## Installation
pytorch 1.7.1
python 3.7.3

## Datasets

Download the CIFAR10 from http://www.cs.toronto.edu/~kriz/cifar.html or you can get it from https://pan.baidu.com/s/1ogAFopdVzswge2Aaru_lvw (code: k5v8), creat data floder and unzip the cifar-10-python.tar.gz under './data'. 

## Pre_trained model

You can download the pretrained file from https://pan.baidu.com/s/1CuUj-XIXwecxWMEcLoJzPg (code: ox9n), creat Vit_weights floder and pretrained file under ./Vit_weights 

## Train
```
python main.py 
```
## Result

Base on the pretrained weight, after one epoch, we get 98.1 Accuracy

model  | dataset  | acc
---- | ----- | ------  
ViT-B_16  | CIFAR10 | 98.1 
