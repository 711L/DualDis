# -*- coding: utf-8 -*-            
# @Author : wobhky
# @File : vis_retrieval_capability.py
# @Time : 2025/6/8 20:19
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
import os
from PIL import Image
import random
def vis_res():
    with open('xxx4veri776.txt', 'r') as f:
        all = f.readlines()
    lines = []
    for line in all:
        if line.strip().split(' ')[0] in special:
            lines.append(line)
    print('res_list:',lines)
    lines = all
    random.shuffle(lines)

    fig = plt.figure(figsize=(20, 12))
    for i in range(5):
        line = lines[i]
        print('row {}'.format(i),line)
        res_list = line.strip().split(' ')
        for j in range(11):
            ax = fig.add_subplot(5, 11, i*11+j+1)  

            if j == 0:
                img = Image.open(os.path.join('/home/xxxx/mypywork/CV/MyVID/datas/VeRi/image_query',res_list[j]+'.jpg'))
    
            else:
                img = Image.open(os.path.join('/home/wangfeiyu/mypywork/CV/MyVID/datas/VeRi/image_test', res_list[j] +'.jpg'))
            img_resized = img.resize((288, 288))
            img = np.array(img_resized)
            ax.imshow(img)
            if j != 0:
                print(res_list[j].split('_')[0],res_list[0].split('_')[0])
                if res_list[j].split('_')[0] == res_list[0].split('_')[0]:
                    for spine in ['top', 'bottom', 'left', 'right']:
                        ax.spines[spine].set_color('green')
                        ax.spines[spine].set_linewidth(2)
                else:
                    for spine in ['top', 'bottom', 'left', 'right']:
                        ax.spines[spine].set_color('red')
                        ax.spines[spine].set_linewidth(2)

            ax.set_xticks([])
            ax.set_yticks([])
            #ax.axis('off')
    fig.subplots_adjust(wspace=0.1, hspace=0.5)  # 调整水平和垂直间距
    plt.savefig(
        "sorting.pdf",
        dpi=300,  
        bbox_inches="tight",  
        facecolor="white",  
        transparent=False 
    )
    plt.show()

vis_res()
