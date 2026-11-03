import timm
import torch
import torch.nn as nn

def get_vit_model(num_classes=38):
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    return model