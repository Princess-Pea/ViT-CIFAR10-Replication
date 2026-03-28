import timm
import torch.nn as nn

def create_vit_model(num_classes=10):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
    return model