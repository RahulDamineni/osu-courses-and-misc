import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter

EXPT_NAME = "rnn_vanialla"
BATCH_SIZE = 16
NUM_EPOCHS = 10000
LEARNING_RATE = 0.001
NUM_HIDDEN_UNITS = 2
NUM_LAYERS = 1


class AirPassengersDataset(torch.utils.data.Dataset):

    def __init__(self,
                 path_to_csv,
                 train=True,
                 train_per=0.75):

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


class LSTM(nn.Module):

    def __init__(self, input_token_len, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_token_len
        self.hidden_size = hidden_size
        seq_length = 1

        self.lstm = nn.RNN(
            input_size=input_token_len * seq_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # (batch, seq_len, input_size)
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Propagate input through LSTM
        out, h_out = self.lstm(x, h_0)

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


writer = SummaryWriter(log_dir=f'./log/{EXPT_NAME}')

train = AirPassengersDataset(path_to_csv="./AirPassengers.csv", train=True)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE)

test = AirPassengersDataset(path_to_csv="./AirPassengers.csv", train=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE)

lstm = LSTM(
    input_token_len=1,
    hidden_size=NUM_HIDDEN_UNITS,
    num_layers=NUM_LAYERS
)
lstm.train()
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

        train_loss = criterion(y_hat, y)
        train_loss.backward()

        optimizer.step()

    if e % 500 == 0:
        # Evaluate
        lstm.eval()

        test_x = torch.Tensor(test.x.values).float().view(-1, 1, 1)
        test_y = torch.Tensor(test.y.values).float().view(-1, 1)
        pred_y = lstm.forward(test_x)

        test_loss = criterion(pred_y, test_y)

        # logging
        print(f'training: EPOCH: {e}, loss: {train_loss}')
        print(f'testing: EPOCH: {e}, loss: {test_loss}')
        writer.add_scalars("rnn_vanialla/loss", {
            "test": test_loss.item(),
            "train": train_loss.item()
        }, e)

        lstm.train()


# Predictions
tr_x = torch.Tensor(train.x.values).float().view(-1, 1, 1)
test_x = torch.Tensor(test.x.values).float().view(-1, 1, 1)
train_forecast = lstm.forward(x=tr_x).view(-1)
test_forecast = lstm.forward(x=test_x).view(-1)
all_forecast = train_forecast.tolist() + test_forecast.tolist()
for id, tr_f in enumerate(all_forecast, 1):
    writer.add_scalar("rnn_vanialla/Predictions", tr_f, id)
