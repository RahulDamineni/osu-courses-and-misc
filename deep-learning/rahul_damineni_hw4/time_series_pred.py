import pandas as pd
import torch
import torch.nn as nn


class AirPassengersDataset(torch.utils.data.Dataset):

    def __init__(self,
                 path_to_csv,
                 train=True,
                 train_per=0.75,
                 normalize=False):

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

    def normalizer(self):


class LSTM(nn.Module):

    def __init__(self, input_token_len, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_token_len
        self.hidden_size = hidden_size
        seq_length = 1

        self.lstm = nn.LSTM(
            input_size=input_token_len * seq_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # (batch, seq_len, input_size)
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


BATCH_SIZE = 10
NUM_EPOCHS = 10000
LEARNING_RATE = 0.001
NUM_HIDDEN_UNITS = 2
NUM_LAYERS = 1

denormalizer, train = AirPassengersDataset(
    path_to_csv="./AirPassengers.csv",
    train=True,
    normalize=True
)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE)

test = AirPassengersDataset(
    path_to_csv="./AirPassengers.csv",
    train=False,
    normalize=False
)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE)

lstm = LSTM(
    input_token_len=1,
    hidden_size=NUM_HIDDEN_UNITS,
    num_layers=NUM_LAYERS
)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)


for e, _ in enumerate(range(NUM_EPOCHS), 1):
    for i, data in enumerate(train_loader):

        x, y = data["x"].float(), data["y"].float()
        # print(f'Mini-batch: {i} | X: {x.size()}, Y: {y.size()}')
        x = x.view(-1, 1, 1)  # (batch, seq_len, input_size)
        y = y.view(-1, 1)  # (batch, output_size)
        y_hat = lstm.forward(x=x)

        optimizer.zero_grad()

        loss = criterion(y_hat, y)
        loss.backward()

        optimizer.step()

    if e % 500 == 0:
        print(f'EPOCH: {e}, loss: {loss}')
