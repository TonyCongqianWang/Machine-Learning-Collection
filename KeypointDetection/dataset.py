import pandas as pd
import numpy as np
import config
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


class FacialKeypointDataset(Dataset):
    def __init__(self, data, train=True, transform=None):
        super().__init__()
        self.category_names = ["0_x","0_y", "1_x", "1_y", "2_x", "2_y", "3_x", "3_y", "4_x", "4_y", "5_x", "5_y"]
        self.transform = transform
        self.train = train
        if isinstance(data, str):
            data = pd.read_csv(data)
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.train:
            image = np.array(self.data.iloc[index, 12].split()).astype(np.float32)
            labels = np.array(self.data.iloc[index, :12].tolist())
            labels[np.isnan(labels)] = -1
        else:
            image = np.array(self.data.iloc[index, 1].split()).astype(np.float32)
            labels = np.zeros(12)

        ignore_indices = labels == -1
        labels = labels.reshape(6, 2)

        if self.transform:
            image = np.repeat(image.reshape(96, 84, 1), 3, 2).astype(np.uint8)
            augmentations = self.transform(image=image, keypoints=labels)
            image = augmentations["image"]
            labels = augmentations["keypoints"]

        labels = np.array(labels).reshape(-1)
        labels[ignore_indices] = -1

        return image, labels.astype(np.float32)


if __name__ == "__main__":
    data = pd.read_csv("data/train_4.csv")

    ds = FacialKeypointDataset(data=data, train=True, transform=config.train_transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    for idx, (x, y) in enumerate(loader):
        plt.imshow(x[0][0].detach().cpu().numpy(), cmap='gray')
        plt.plot(y[0][0::2].detach().cpu().numpy(), y[0][1::2].detach().cpu().numpy(), "go")
        plt.show()
