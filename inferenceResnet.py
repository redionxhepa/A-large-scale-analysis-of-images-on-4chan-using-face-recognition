



import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim







weightsPath='/home/redion/redion_files/AnimeClassifier/scripts/savedmodels/3weights.h5'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#recreate the model
model = models.resnet18(pretrained=False).to(device)
model.fc = nn.Sequential(
               nn.Linear(512, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)
model.load_state_dict(torch.load(weightsPath))


#data tranforms need to load the images
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}




validation_img_paths = ["/home/redion/redion_files/AnimeClassifier/randomRelatedStuff/animeSample.png",
                              "/home/redion/redion_files/AnimeClassifier/randomRelatedStuff/hillary.jpeg", "/home/redion/redion_files/AnimeClassifier/randomRelatedStuff/onlyfrog.jpeg",
                          "/home/redion/redion_files/AnimeClassifier/randomRelatedStuff/twoperson1.jpeg" , "/home/redion/redion_files/AnimeClassifier/randomRelatedStuff/twoperson3.jpeg"]
img_list = [Image.open(img_path) for img_path in validation_img_paths]


validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                for img in img_list])
#model.eval()
pred_logits_tensor = model(validation_batch)

pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()


# to be changed

for i, img in enumerate(img_list):
  print("Image:"+str(validation_img_paths[i])+ "{:.0f}% Anime, {:.0f}% Person".format(100*pred_probs[i,0],
                                                            100*pred_probs[i,1]))
