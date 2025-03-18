#This is the main file that will take a trained model checkpoint and carry out required analysis on it.

import os
import sys
import argparse
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
import pdb
from vizMat import visualize_linear_sparsity
from sparsity import get_layerwise_sparsity
from ood import evaluate_ood, parse_args
from mcnemarsTest import evaluatepval

#Command line arguments to run the analysis
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', metavar='MODEL', help='path to model checkpoint')
#parser.add_argument('--threshold', default=1e-3, type=float, help='threshold for pruning')
#parser.add_argument('--compare', default='', type=str, help='path to model checkpoint to compare with')
args = parser.parse_args()

#Create a folder named analysis in the same directory as the checkpoint to store the analysis results
analysis_dir = os.path.join(os.path.dirname(args.model), 'analysis')
os.makedirs(analysis_dir, exist_ok=True)

#Visualize the sparsity pattern of the model
#checkpoint = torch.load(args.model, map_location='cpu')
#model = checkpoint['state_dict']
#visualize_linear_sparsity(model, analysis_dir)

#Get the sparsity level of each layer in the model and save that to a file with columns: name, dimensions, sparsity
#output_file = os.path.join(analysis_dir, 'sparsity.csv')
#get_layerwise_sparsity(model, output_file)

#get the ood performance
""" from argparse import Namespace
args = Namespace(
        model='deit_tiny_patch16_224',
        checkpoint=args.model,
        data_dir='/p/dataset/cifar-10-batches-py',
        batch_size=128,
        workers=4,
        device='cuda',
        channels_last=False,
        pin_mem=False,
        num_classes=10,
        sparsity=0.80,
        sparsityType='permkm',
        nm_n=4,
        nm_m=20,
        block_size=2,
        round_permutations=False,
)
    
metrics = evaluate_ood(args)
print("Returned Metrics:")
print(f"Loss: {metrics['loss']:.4f}")
print(f"Top-1 Accuracy: {metrics['top1']:.2f}%")
print(f"Top-5 Accuracy: {metrics['top5']:.2f}%")
 """
#Do McNemar's Test
from argparse import Namespace
args = Namespace(
        val_split='validation',
        dataset='imagenet',
        modelA='deit_tiny_patch16_224',
        checkpointA='/localdisk/Abhishek/pytorch-image-models_perm/output/train/deit-tiny-diagI100_patchLinear_autoshuffle_4090_0.80/model_best.pth.tar',
        modelB='deit_tiny_patch16_224',
        checkpointB='/localdisk/Abhishek/pytorch-image-models_perm/output/train/deit-tiny-diagI100_patchLinear_autoshuffle_4090_0.98/model_best.pth.tar',
        data_dir='/p/dataset/ImageNet100',
        batch_size=128,
        workers=4,
        device='cuda',
        channels_last=False,
        pin_mem=False,
        num_classes=1000,
        sparsity=0.80,
        sparsityType='permkm',
        nm_n=4,
        nm_m=20,
        block_size=2,
        round_permutations=False,
)
evaluatepval(args)