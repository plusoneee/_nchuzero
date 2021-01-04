
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class SimpleModel(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input = nn.Linear(input_size, 10)
        self.hidden_layer = nn.Linear(10, 20)
        self.output = nn.Linear(20, output_size)

    def forward(self, x):
        out = self.input(x)
        out = self.hidden_layer(out)
        out = self.output(out)
        return out

def build_toy_dataset():
    X = [i for i in range(11)]

    X_train = np.array(X, dtype=np.float32).reshape(-1, 1) # 2D required
    print(' X_train shape:', X_train.shape)

    # y = 2X + 1
    y = [2 * i + 1 for i in X]
    y_train = np.array(y, dtype=np.float32).reshape(-1, 1)  # 2D required
    print(' y_train shape:', y_train.shape)
    return (X_train, y_train)


def plot_predict(X_train, y_train, y_prediction):
    plt.clf() # clear figure

    # plat true data
    plt.plot(X_train, y_train, 'go', label='True Data', alpha=0.5)

    # plat true data
    plt.plot(X_train, y_prediction, '--', label='True Data', alpha=0.5)

    # legend and plot
    plt.legend(loc='best')
    plt.show()

@hydra.main(config_name='config')
def run_my_simple_net(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))

    model_cfg = cfg.model

    assert model_cfg.input_size == 1, "model input size should be 1"

    model = SimpleModel(model_cfg.input_size, model_cfg.output_size)
    print('Model Summary\n', model)

    # loss function
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=model_cfg.learning_rate)

    X_train, y_train = build_toy_dataset()

    for epoch in range(model_cfg.epochs):
        epoch += 1 # for print

        inputs = torch.from_numpy(X_train)
        labels = torch.from_numpy(y_train)

        # clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # forward to get output
        outputs = model(inputs)  # output is y and inputs in x
        # calculate loss
        loss = criterion(outputs, labels)
        # getting gradients w.r.t. parameters
        loss.backward()
        # updating parameters
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.data))

    # predition
    y_prediction = model(torch.from_numpy(X_train)).data.numpy()

    if model_cfg.plot:
        plot_predict(X_train, y_train, y_prediction)

    if model_cfg.save.need:
        torch.save(model.state_dict(), model_cfg.save.name)

if __name__ == "__main__":
    run_my_simple_net()