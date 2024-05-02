import torch
from torch import nn

class NoiseModel(nn.Module):
    def __init__(self, n_steps: int):
        super(NoiseModel, self).__init__()
        self.n_steps = n_steps
        w = 16
        t_w = 1
        self.t_layer = nn.Sequential(nn.Linear(1, t_w),
            nn.ReLU(),
            nn.Linear(t_w, t_w),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(nn.Linear(t_w + 2, w),
            nn.ReLU(),
            nn.Linear(w, w),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(nn.Linear(w, w),
            nn.ReLU(),
            nn.Linear(w, w),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(nn.Linear(w + t_w, w),
            nn.ReLU(),
            nn.Linear(w, w),
            nn.Tanh()
        )
        self.last_layer = nn.Linear(w, 2)

    def forward(self, x, t):
        t = (t.float() / self.n_steps) - 0.5
        temb = self.t_layer(t)

        output = self.layer1(torch.concat([x, temb], axis=-1))
        output = self.layer2(output)
        output = self.layer3(torch.concat([output, temb], axis=-1))
        return self.last_layer(output)


class DiffusionModel:
    def __init__(self, model: nn.Module, n_steps: int, beta_1: float, beta_t: float, device: str):
        """
            model: model used to predict noise of diffused images
            n_steps: total number of diffusion steps
            beta_1: initial beta for beta scheduler
            beta_t: last beta for beta scheduler
            device: torch device to load tensors and model on
        """
        self.model = model.to(device)
        self.n_steps = n_steps
        self.betas = torch.linspace(beta_1, beta_t, self.n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, axis=0)

        self.r_alphas_bar = torch.sqrt(self.alphas_bar)
        self.r_1m_alphas_bar = torch.sqrt(1 - self.alphas_bar)

        self.inv_r_alphas = torch.pow(self.alphas, -0.5)
        self.pre_noise_terms = self.betas / self.r_1m_alphas_bar
        self.sigmas = torch.pow(self.betas, 0.5)

        self.device = device

    def diffuse(self, x, t):
        eps = torch.randn(x.shape).to(self.device)
        t = t - 1
        diffused = self.r_alphas_bar[t] * x + self.r_1m_alphas_bar[t] * eps
        return eps, diffused

    def denoise(self, x: torch.Tensor, t: torch.Tensor):
        """
        Denoise random samples x for t steps.
        x: initial 2d data points to denoise
        t: number of denoising time steps
        return (denoised data points, list of each denoised data points for all diffusion steps)
        """
        n_samples = 1
        if len(x.shape)>1:
            n_samples = x.shape[0]
        all_x = [x]
        for i in range(t, 0, -1):
            z = torch.randn(x.shape).to(self.device)
            if i == 1:
                z = z * 0
            steps = torch.full((n_samples,), i, dtype=torch.int, device=self.device).unsqueeze(1)
            model_output = self.model(x, steps)
            x = self.inv_r_alphas[i - 1] * (x - self.pre_noise_terms[i - 1] * model_output) + self.sigmas[i - 1] * z
            # x = (self.r_alphas_bar[i - 1] / self.r_alphas_bar[i]) * (x - self.r_1m_alphas_bar[i] * model_output) + self.r_1m_alphas_bar[i-1] * model_output  
            all_x.append(x)
        return x, all_x