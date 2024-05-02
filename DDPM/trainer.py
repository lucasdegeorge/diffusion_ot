import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
from DDPM.model import DiffusionModel, NoiseModel
import matplotlib.pyplot as plt


class DiffusionTrainer:
    def __init__(self, datapoints, batch_size, beta_1, beta_t, n_steps, lr, device):
        self.model = NoiseModel(n_steps)
        self.diffuser = DiffusionModel(self.model, n_steps, beta_1, beta_t, device)
        self.optimizer = torch.optim.Adam(self.diffuser.model.parameters(), lr=lr, weight_decay=0.001)
        dataset = TensorDataset(torch.Tensor(datapoints))
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
        self.device = device
    
    def train(self, n_epochs):
        self.diffuser.model.train()
        all_losses = []

        for epoch in tqdm(range(n_epochs)):
            for batch in self.dataloader:
                batch = batch[0].to(self.device)
                t = torch.randint(1, self.diffuser.n_steps + 1, (len(batch),)).unsqueeze(1).to(self.device)
                eps, diffused = self.diffuser.diffuse(batch, t)
                pred_eps = self.model(diffused, t)

                loss = (eps - pred_eps) ** 2
                loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                loss = loss.detach().item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()
                all_losses.append(loss)

        return all_losses


if __name__=='__main__':
    # Hyper parameters
    batch_size = 1024
    n_epochs = 100
    n_steps = 100
    lr = 0.003
    device = "cpu"
    beta_1 = 0.0001
    beta_t = 0.02

    # Instantiation
    datapoints = generate_datapoints(n_samples=10000)
    trainer = DiffusionTrainer(datapoints, batch_size, beta_1, beta_t, n_steps, lr, device)
    all_losses = trainer.train(n_epochs)
    plt.figure()
    plt.plot(all_losses)
    plt.show()

