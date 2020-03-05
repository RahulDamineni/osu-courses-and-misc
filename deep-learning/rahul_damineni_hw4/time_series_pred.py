import pandas as pd
import torch


class AirPassengersDataset(torch.utils.data.Dataset):

    def __init__(self, path_to_csv, train=True, train_per=0.75):

        df = pd.read_csv(path_to_csv)

        train_count = int(0.75 * len(df))
        if train is True:
            df = df[:train_count]
        else:
            df = df[train_count:]

        self.x = df.AirPassengers
        self.y = df.AirPassengers.shift(periods=-1, fill_value=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "x": self.x.iloc[idx],
            "y": self.y.iloc[idx]
        }

        return sample


BATCH_SIZE = 10
train = AirPassengersDataset(path_to_csv="./AirPassengers.csv", train=True)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE)

test = AirPassengersDataset(path_to_csv="./AirPassengers.csv", train=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE)

for i, data in enumerate(train_loader):

    x, y = data["x"], data["y"]
    print(f'Mini-batch: {i} | X: {x.size()}, Y: {y.size()}')
