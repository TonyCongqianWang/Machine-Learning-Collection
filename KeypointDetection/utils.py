import torch
import numpy as np
import config
import pandas as pd
from tqdm import tqdm
from dataset import FacialKeypointDataset
from torch.utils.data import DataLoader


def get_submission(dataset_path, model_4):
    """
    This can be done a lot faster.. but it didn't take
    too much time to do it in this inefficient way
    """
    data = pd.read_csv(dataset_path)
    
    test_ds = FacialKeypointDataset(
        data=data,
        transform=config.val_transforms,
        train=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )

    model_4.eval()
    category_names = test_ds.category_names

    imageIds = {"ImageId" : data["ImageId"]}
    predictions = {cat_name : [] for cat_name in category_names}

    for image in tqdm(test_loader):
        image = image.to(config.DEVICE)
        preds_4 = model_4(image).squeeze(0)

        for cat_idx, cat_name in enumerate(category_names):
            predictions[cat_name].append(preds_4[cat_idx].item())
            
    df = pd.DataFrame({**imageIds, **predictions})
    df.to_csv("data/submission.csv", index=True)
    model_4.train()


def get_rmse(loader, model, loss_fn, device):
    model.eval()
    num_examples = 0
    losses = []
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = loss_fn(scores[targets != -1], targets[targets != -1])
        num_examples += scores[targets != -1].shape[0]
        losses.append(loss.item())

    model.train()
    return (sum(losses)/num_examples)**0.5

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr