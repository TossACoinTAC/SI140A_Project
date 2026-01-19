import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import math
import matplotlib.pyplot as plt

# ===== Configuration =====
DATA_PATH = os.path.join("Database", "Wechat_Samples.json")
MODEL_DIR = "DiffusionModel"

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 1000
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 5
HIDDEN_DIM = 128


# ===== Dataset and Data Loading =====
class WechatDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        self.data = torch.tensor(data, dtype=torch.float32)

        # Keep only first 5 dimensions
        if self.data.shape[1] > 5:
            self.data = self.data[:, :5]

        # Normalization
        self.mean = self.data.mean(dim=0)
        self.std = self.data.std(dim=0)
        self.normalized_data = (self.data - self.mean) / (self.std + 1e-8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.normalized_data[idx]


# ===== Model Components =====
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Main network
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.mid_layers = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.final_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        # Embed time
        t_emb = self.time_mlp(t)

        # Embed input
        x_emb = self.input_layer(x)

        # Concatenate x embedding and time embedding
        # Note: We concatenate them or add them. Concatenating to input of interaction layers is common.
        # Here we concat along feature dimension
        h = torch.cat([x_emb, t_emb], dim=1)

        h = self.mid_layers(h)
        return self.final_layer(h)


# ===== Training Logic =====
def get_beta_schedule(timesteps, start, end):
    return torch.linspace(start, end, timesteps)


def train():
    print(f"Using device: {DEVICE}")

    # 1. Prepare Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Database file not found at {DATA_PATH}")
        return

    dataset = WechatDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset)} samples.")

    # 2. Diffusion Setup
    betas = get_beta_schedule(TIMESTEPS, BETA_START, BETA_END).to(DEVICE)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    # 3. Model Setup
    model = DiffusionModel(INPUT_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # 4. Training Loop
    model.train()
    loss_history = []
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(DEVICE)

            # Sample t uniformly
            t = torch.randint(0, TIMESTEPS, (batch.size(0),), device=DEVICE).long()

            # Sample noise
            noise = torch.randn_like(batch)

            # Forward diffusion q(x_t | x_0)
            # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
            sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None]
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])[:, None]

            x_t = sqrt_alpha_cumprod_t * batch + sqrt_one_minus_alpha_cumprod_t * noise

            # Predict noise
            noise_pred = model(x_t, t)

            # Loss
            loss = criterion(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # 5. Save Model and Stats
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "diffusion_model.pth"))
    # Save normalization stats for generation later
    stats = {"mean": dataset.mean.tolist(), "std": dataset.std.tolist()}
    with open(os.path.join(MODEL_DIR, "model_stats.json"), "w") as f:
        json.dump(stats, f)

    print("Training complete. Model saved to DiffusionModel/diffusion_model.pth")

    # 6. Visualize Loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Evolution of Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, "loss_evolution.png"))
    print("Loss plot saved to DiffusionModel/loss_evolution.png")


if __name__ == "__main__":
    train()
