# - model.py - #
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch

class SRModels(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet('se_resnext50_32x4d', in_channels=3, classes=3, activation='sigmoid',
        encoder_weights='imagenet')

    def forward(self, x):
        output = self.model(x)
        output = output.permute(0,2,3,1) # (bs, c, h, w) --> (bs, h, w c)

        return output

class SRModelsIf(nn.Module):
    def __init__(self):
        super().__init__()
        # go down
        self.model = smp.Unet('se_resnext50_32x4d', in_channels=3, classes=3, activation='sigmoid')
        self.model2 = smp.Unet('se_resnext50_32x4d', in_channels=3, classes=3, activation='sigmoid')
        # weight
        m_p = './saved_models/41e_32.1909_s.pth'
        m_p2 = './saved_models/46e_31.9065_s.pth'
        a = torch.load(m_p)
        b = torch.load(m_p2)
        new_weight = {}
        for name_ in a.keys():
            new_weight[name_.split('module.model.')[-1]] = a[name_]
            
        new_weight2 = {}
        for name_ in a.keys():
            new_weight2[name_.split('module.model.')[-1]] = b[name_]
            
        self.model.load_state_dict(new_weight, strict=True)
        self.model2.load_state_dict(new_weight2, strict=True)
        
            


    def forward(self, x):
        output = (self.model(x) + self.model2(x))/2
        output = output.permute(0,2,3,1) # (bs, c, h, w) --> (bs, h, w c)

        return output
