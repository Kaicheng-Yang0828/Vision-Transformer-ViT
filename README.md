# Vit-ImageClassification

## Introduction
This project uses ViT to perform image classification tasks on DATA set CIFAR10. The implement of Vit and pretrained weight are from https://github.com/asyml/vision-transformer-pytorch. 

![The architecture of ViT](https://github.com/Kaicheng-Yang0828/Vit-ImageClassification/blob/main/pic/VIT.png)

## Installation
Create environment:
```
conda create --name vit --file requirements.txt
conda activate vit
```

## Datasets

Download the CIFAR10 from http://www.cs.toronto.edu/~kriz/cifar.html, creat data floder and unzip the cifar-10-python.tar.gz in 'data/'. 

# Fine-Tune/Train
```
python main.py 
```
