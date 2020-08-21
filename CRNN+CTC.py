import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from dataloader import Imgdataset
from FCSRN import mnist_net

num_epoch=10

train_address = "./SCUT_WMN_DataSet"
train_dataset = Imgdataset(train_address)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=100,shuffle=True)

# for i in range(100):
#     train_dataset.__getitem__(i)

network=mnist_net()

optimizer = optim.Adam(network.parameters(), lr=0.01)
loss_fn=nn.CrossEntropyLoss()

for epoch in range(num_epoch):
    print("epoch: ",epoch," ")
    total_loss=0
    total_correct=0
    for batch in train_loader:
        images,labels=batch
        labels=list(labels[0])
        print(labels)
        for i in range(len(labels)):
            labels[i]=float(labels[i])
        labels=torch.FloatTensor(labels)
        labels=labels.unsqueeze(dim=1)
        print("images: ",images.shape)
        print("labels: ",labels.shape)

        preds=network(images)
        print("preds:", preds.shape)
        loss = loss_fn(preds,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("loss_item: ",loss.item())
        total_loss+=loss.item()

    print("total_loss:",total_loss)

torch.save(network,"./network.pth")
