import dataset
import os 
import pandas as pd
import pathlib
import torch
import torch.nn as nn
import trainer
import plotting
from torch.utils.data import DataLoader
import utils
from torchvision import models,transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Use GPU if it's available or else use CPU.
print(device) #Prints the device we're using.

main_folder = pathlib.Path(__file__)
parent_folder= main_folder.parents[1]
data_folder = os.path.join(parent_folder,
                           "kaggle",
                           "input",
                           "aptos2019-blindness-detection")

# window_size=40
# steps=15
# batch_size=64
# hidden_size =512
# lr=0.001
# epochs = 25


# net = 'MLP'

train_df = pd.read_csv(os.path.join(data_folder,"train.csv"))
test_df = pd.read_csv(os.path.join(data_folder,"test.csv"))
print(f'No.of.training_samples: {len(train_df)}')
print(f'No.of.testing_samples: {len(test_df)}')

class_weights = utils.calc_class_weights(train_df,device)
print(class_weights)


image_transform = transforms.Compose([transforms.Resize([16,16]),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) #Transformations to apply to the image.
data_set = dataset.DRDataset(train_df,
                             os.path.join(data_folder, 'train_images'),
                             image_transform=image_transform)
#Split the data_set so that valid_set contains 0.1 samples of the data_set. 
train_set,valid_set = torch.utils.data.random_split(data_set,[3302,360])


train_dataloader = DataLoader(train_set,batch_size=512,shuffle=True) #DataLoader for train_set.
valid_dataloader = DataLoader(valid_set,batch_size=512,shuffle=False) #DataLoader for validation_set.



#Since we've less data, we'll use Transfer learning.
model = models.resnet34(pretrained=True) #Downloads the resnet18 model which is pretrained on Imagenet dataset.

#Replace the Final layer of pretrained resnet18 with 4 new layers.
model.fc = nn.Sequential(nn.Linear(512,256),nn.Linear(256,128),nn.Linear(128,64),nn.Linear(64,5))
model = model.to(device) #Moves the model to the device.


loss_fn   = nn.CrossEntropyLoss(weight=class_weights) #CrossEntropyLoss with class_weights.
optimizer = torch.optim.SGD(model.parameters(),lr=0.001) 
nb_epochs = 30
#Call the optimize function.
train_losses, valid_losses = trainer.optimize(train_dataloader,
                                              valid_dataloader,
                                              model,loss_fn,
                                              optimizer,
                                              nb_epochs,
                                              device)


plotting.loss_plots(nb_epochs, train_losses, valid_losses)


test_set = dataset.DRDataset(test_df,
                             os.path.join(data_folder, 'test_images'),
                             image_transform = image_transform,
                             train = False )
test_dataloader = DataLoader(test_set, batch_size=128, shuffle=False) #DataLoader for test_set.

labels = utils.test(test_dataloader,model) #Calls the test function.