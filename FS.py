# -*- coding: utf-8 -*-            
# @Author : wobhky
# @File : FS.py
# @Time : 2025/7/12 13:50

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from DualDis_model import DualDis_model

img_path = "/home/xxxx/mypywork/CV/MyVID/datas/VeRi/image_test/0582_c004_00030275_0.jpg"

test_transform = transforms.Compose([
            transforms.Resize((288, 288), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image = Image.open(img_path).convert('RGB')
img_tensor = test_transform(image).unsqueeze(0)
img_tensor = img_tensor.to(device)
model = FilterModel(576)
model.load_state_dict(torch.load('/home/xxxx/NextWork/FeatureDecouplingHit/DualDis_veri776_attention4relu/FilterModel/model_xxx.pt'))
model = model.to(device)
model.eval()

with torch.no_grad():
    x = model.post_backbone(img_tensor)
    feature_map = model.layer4(x)
    flipped_feature_map = torch.flip(feature_map, [3])
    distilled, weights, attention, r1, r2, res = model.down(feature_map)

    sym = (feature_map + flipped_feature_map) / 2
    asym = (feature_map - flipped_feature_map)
    sym_part = model.down.sasa.sym_conv(sym) * attention
    asym_part = model.down.sasa.asym_conv(asym) * (1 - attention)
    fused = (feature_map + torch.cat([sym_part, asym_part], dim=1)) 

plt.figure(figsize=(12, 18))  

plt.subplot(6, 4, (1, 4))  
plt.imshow(image)
plt.title("Original Input Image", fontsize=18)
plt.axis('off')

channels = [0, 1000, 1500, 2047]
row_titles = ['Original', 'Flipped', 'Sym', 'Asym', 'Fused']

for row_idx, row_name in enumerate(row_titles):
    for col_idx, ch in enumerate(channels):
        pos = (row_idx + 1) * 4 + col_idx + 1 

        plt.subplot(6, 4, pos)
        if row_name == 'Original':
            plt.imshow(feature_map[0, ch].cpu().numpy(), cmap='hot')
        elif row_name == 'Flipped':
            plt.imshow(flipped_feature_map[0, ch].cpu().numpy(), cmap='hot')
        elif row_name == 'Sym':
            data = ((feature_map[0, ch] + flipped_feature_map[0, ch]) / 2).abs().cpu().numpy()
            plt.imshow(data, cmap="hot")
        elif row_name == 'Asym':
            data = (feature_map[0, ch] - flipped_feature_map[0, ch]).cpu().numpy()
            plt.imshow(data, cmap="hot")
        else:  # Fused
            plt.imshow(fused[0, ch].cpu().numpy(), cmap='hot')

        plt.title(f"{row_name} Ch-{ch}", fontsize=18)
        plt.axis('off')
        plt.colorbar()


plt.tight_layout()
plt.savefig('feature_map_flip_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
