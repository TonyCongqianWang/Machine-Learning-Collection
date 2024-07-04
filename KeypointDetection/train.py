import torch, wandb, argparse
from dataset import FacialKeypointDataset
from torch import nn, optim
import os
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_rmse,
    get_submission
)

def train_one_epoch(loader, model, optimizer, loss_fn, device):
    losses = []
    loop = tqdm(loader)
    num_examples = 0
    for _, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        scores[targets == -1] = -1
        loss = loss_fn(scores, targets)
        num_examples += torch.numel(scores[targets != -1])
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_mse= (sum(losses)/num_examples)**0.5
    return loss_mse

def main():
    
    parser = argparse.ArgumentParser(
    description='convert image dataset to csv format')

    parser.add_argument("dataset", type=str,
                    help='name of dataset located in data/')
    parser.add_argument("--resume", default=True,
                    help='name of dataset located in data/')
    args = parser.parse_args()
    
    CHECKPOINT_FILE = f"models/checkpoints/checkpoint_{args.dataset}.pth.tar"

    wandb.init(
        project="keypoints",
        config={
            "initial_lr" :config.LEARNING_RATE,
        }
    )
    
    train_ds = FacialKeypointDataset(
        data=f"data/{args.dataset}_train.csv",
        transform=config.train_transforms,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
    )
    val_ds = FacialKeypointDataset(
        transform=config.val_transforms,
        data=f"data/{args.dataset}_val.csv",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )

    loss_fn = nn.MSELoss(reduction="sum")
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Linear(1280, 12)
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    if args.resume and CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)

    if config.SUBMISSION_MODEL and config.SUBMISSION_MODEL in os.listdir():
        model_sub = EfficientNet.from_pretrained("efficientnet-b0")
        model_sub._fc = nn.Linear(1280, 12)
        model_sub = model_sub.to(config.DEVICE)
        load_checkpoint(torch.load(CHECKPOINT_FILE), model_sub, optimizer, config.LEARNING_RATE)
        get_submission("data/test.csv", model_sub)

    for epoch in range(config.NUM_EPOCHS):
        print(f"Train epoch: {epoch}")
        get_rmse(val_loader, model, loss_fn, config.DEVICE)
        loss_mse = train_one_epoch(train_loader, model, optimizer, loss_fn, config.DEVICE)
        print(f"Train loss average: {loss_mse}")

        # get on validation
        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            loss_mse = get_rmse(train_loader, model, loss_fn, config.DEVICE)
            save_checkpoint(checkpoint, filename=CHECKPOINT_FILE)
            save_checkpoint(checkpoint, filename=CHECKPOINT_FILE + f"_{args.dataset}_{epoch}_{loss_mse}.pth.tar")
            wandb.log({"loss": loss_mse})
            print(f"Valdation loss: {loss_mse}")

if __name__ == "__main__":
    main()
