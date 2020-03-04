import pandas as pd
import torch

df = pd.read_csv("./AirPassengers.csv")
df = df.sort_values(by=["Month"])

x = df.AirPassengers
y = df.AirPassengers.shift(periods=-1, fill_value=0)


train_count = int(0.75 * len(df))
x_train = torch.tensor(x[:train_count].values)
y_train = torch.tensor(y[:train_count].values)
x_test = torch.tensor(x[train_count:].values)
y_test = torch.tensor(y[train_count:].values)

# import ipdb
# ipdb.set_trace()
train = torch.utils.data.TensorDataset(x_train, y_train)
test = torch.utils.data.TensorDataset(x_test, y_test)

BATCH_SIZE = 10
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE)

for i, data in enumerate(train_loader):

    x, y = data
    print(f'Mini-batch: {i} | X: {x.size()}, Y: {y.size()}')
