import torch
import torchvision
from torch import nn

def create_effnetb2_model(num_classes:int=3,
                          seed:int=33):

  effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT

  transforms = effnetb2_weights.transforms()

  model = torchvision.models.efficientnet_b2(weights=effnetb2_weights)

  for param in model.parameters():
    param.requires_grad = False

  torch.manual_seed(seed)

  model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                    nn.Linear(in_features=1408,
                                              out_features=num_classes))

  return model, transforms
