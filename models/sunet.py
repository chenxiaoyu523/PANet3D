"""
Encoder for few shot segmentation (UNet)
"""

import torch
import torch.nn as nn
import sparseconvnet as scn

# Options
m = 16 # 16 or 32
residual_blocks=False #True or False
block_reps = 1 #Conv block repetition factor: 1 or 2

dimension=3
full_scale=4096 #Input field size

class UNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(dimension,full_scale, mode=4)).add(
           scn.SubmanifoldConvolution(dimension, 3, m, 3, False)).add(
               scn.FullyConvolutionalNet(dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks)).add(
           scn.BatchNormReLU(448)).add(
           scn.OutputLayer(dimension))
        self.linearx = nn.Linear(448, 448)

        for _, para in enumerate(self.sparseModel.parameters()):
            para.requires_grad = False
        #self.drop = nn.Dropout(0.3)
    def forward(self,x):
        x=self.sparseModel(x)
        x=self.linearx(x)

        return x
