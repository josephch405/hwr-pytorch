import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import data
import model

transforms = data.MakeSquareAndTargets()
dataset = data.HWRSegmentationDataset(
    "data/csv/forms", "data/forms", transforms)

batch_size = 8
validation_split = .2
random_seed = 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=8,
                          num_workers=4, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=8,
                        num_workers=4, sampler=val_sampler)

MAX_EPOCHS = 20

net = model.GreyNet()  # .cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def weighted_loss(pred, target):
    # loss is scaled to .1 for "falsy" grid squares and 100 for "truthy"
    mask = (target[:, 4] == 0)[:, None].float() * .1
    mask += (target[:, 4] > 0)[:, None].float() * 10

    loss = nn.functional.soft_margin_loss

    loss = nn.functional.mse_loss(pred, target, reduction='none')

    loss *= mask
    return loss.mean()


for epoch in range(MAX_EPOCHS):
    running_loss = 0.0
    print(f"epoch {epoch}")
    for i_batch, sample_batched in enumerate(train_loader):
        image = sample_batched['image']  # .cuda()
        target = sample_batched['target'].float()  # .cuda().float()
        optimizer.zero_grad()

        preds = net(image)
        loss = weighted_loss(preds, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i_batch % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i_batch + 1, running_loss / 20))
            running_loss = 0.0

torch.save({
    # 'epoch': epoch,
    'model_state_dict': model.state_dict()
}, "checkpoint")
