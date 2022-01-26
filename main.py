# -*- coding: utf-8 -*-
# @File : main.py
# @Author : Kaicheng Yang
# @Time : 2022/01/26 11:03:50
import argparse
from torchvision import datasets, transforms
import torch
from torchvision.transforms import Resize, ToTensor, Normalize
from PIL import Image
from train import train_model


def main():
    parser = argparse.ArgumentParser()
    # Optimizer parameters
    parser.add_argument("--learning_rate", default = 2e-5, type = float,
                        help = "The initial learning rate for Adam.5e-5")
    parser.add_argument('--opt-eps', default = None, type = float, metavar = 'EPSILON',
                        help = 'Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument("--beta1", type = float, default = 0.99, help = "Adam beta 1.")
    parser.add_argument("--beta2", type = float, default = 0.99, help = "Adam beta 2.")
    parser.add_argument("--eps", type = float, default = 1e-6, help = "Adam epsilon.")
    parser.add_argument('--momentum', type = float, default = 0.9, metavar = 'M',
                        help = 'Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type = float, default = 2e-5,
                        help = 'weight decay (default: 2e-5)')
    parser.add_argument(
        "--warmup", type = int, default = 500, help = "Number of steps to warmup for."
    )
    parser.add_argument("--batch_size", type = int, default = 64, help = "Number of steps to warmup for.")
    parser.add_argument("--epoches", type = int, default = 5, help = "Number of steps to warmup for.")
    #Vit params
    parser.add_argument("--output", default = './output', type = str)
    parser.add_argument("--vit_model", default = './Vit_weights/imagenet21k+imagenet2012_ViT-B_16-224.pth', type = str)
    parser.add_argument("--image_size", type = int, default = 224, help = "input image size", choices = [224, 384])
    parser.add_argument("--num-classes", type = int, default = 10, help = "number of classes in dataset")
    parser.add_argument("--patch_size", type = int, default = 16)
    parser.add_argument("--emb_dim", type = int, default = 768)
    parser.add_argument("--mlp_dim", type = int, default = 3072)
    parser.add_argument("--num_heads", type = int, default = 12)
    parser.add_argument("--num_layers", type = int, default = 12)
    parser.add_argument("--attn_dropout_rate", type = float, default = 0.0)
    parser.add_argument("--dropout_rate", type = float, default = 0.1)

    args = parser.parse_args()
    normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose([
            Resize((224, 224)),
            ToTensor(),
            normalize,
        ])

    trainset = datasets.CIFAR10(root = './data', train = True,
                                        download = False, transform = transform)
                                        
    testset = datasets.CIFAR10(root = './data', train = False,
                                        download = False, transform = transform)
                                        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size,
                                            shuffle = True, num_workers = 2)
                                            
    testloader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size,
                                            shuffle = False, num_workers = 2)
    
    train_model(args, trainloader, testloader)



if __name__ == '__main__':
    main()
    


    