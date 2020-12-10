from torchvision import models
import torch
import os
import numpy as np
from torchvision import transforms
def expand_greyscale(t):
    return t.expand(3, -1, -1)
test_transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(expand_greyscale)
        ])
resnet=models.resnet18(pretrained=True).cuda()
checkpoint = torch.load('checkpoint/byol_check_out.pt')
resnet.load_state_dict(checkpoint)
resnet.eval()
test_path=''
list_img_tensor=[]
for i in range(os.listdir(test_path)):
    img_path=test_path+'/'+str(i+1)+'.png'
    list_img_tensor.append(test_transform(Image.open(path).convert('RGB')))

img_tensors=torch.stack(list_img_tensor)
