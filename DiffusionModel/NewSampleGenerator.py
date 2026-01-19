import torch
import json
import os
import sys

# Add the current directory to sys.path to allow importing from DiffusionModelTrainer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from DiffusionModelTrainer import (
        DiffusionModel,
        get_beta_schedule,
        TIMESTEPS,
        BETA_START,
        BETA_END,
        DEVICE,
        INPUT_DIM,
        HIDDEN_DIM,
    )
except ImportError:
    # Fallback for when running from root as a module or similar issues
    from DiffusionModel.DiffusionModelTrainer import (
        DiffusionModel,
        get_beta_schedule,
        TIMESTEPS,
        BETA_START,
        BETA_END,
        DEVICE,
        INPUT_DIM,
        HIDDEN_DIM,
    )

# ========= Path Configuration =========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "DiffusionModel", "diffusion_model.pth")
STATS_PATH = os.path.join(BASE_DIR, "DiffusionModel", "model_stats.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "Database", "DDPM_Generated_Samples.json")

# ========= Generate Samples =========
NUM_SAMPLES = 30000


def sample(model, num_samples):
    model.eval()
    with torch.no_grad():
        # 1. Prepare betas and alphas
        # Ensure we use the same schedule as training
        betas = get_beta_schedule(TIMESTEPS, BETA_START, BETA_END).to(DEVICE)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        # 2. Initialize x_T with random noise
        # Start from pure Gaussian noise
        x = torch.randn((num_samples, INPUT_DIM)).to(DEVICE)

        print(
            f"Starting sampling process for {num_samples} samples with {TIMESTEPS} timesteps..."
        )

        # 3. Iterative denoising
        for t in reversed(range(TIMESTEPS)):
            # Progress indicator
            if t % 100 == 0:
                print(f"Time step: {t}")

            # Create a batch of time steps
            t_tensor = torch.full((num_samples,), t, device=DEVICE, dtype=torch.long)

            # Predict noise using the model
            predicted_noise = model(x, t_tensor)

            # Get coefficients for this time step
            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            beta = betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # Compute x_{t-1} using the reverse diffusion formula
            # x_{t-1} = (1/sqrt(alpha)) * (x_t - ((1-alpha)/sqrt(1-alpha_cumprod)) * eps) + sigma * z

            inv_sqrt_alpha = 1 / torch.sqrt(alpha)
            noise_coeff = (1 - alpha) / torch.sqrt(1 - alpha_cumprod)
            sigma = torch.sqrt(beta)  # Using sigma_t^2 = beta_t

            x = inv_sqrt_alpha * (x - noise_coeff * predicted_noise) + sigma * noise

    return x


def generate_samples(num_samples=NUM_SAMPLES, output_file=OUTPUT_PATH):
    print(f"Using device: {DEVICE}")
    print(f"Loading model from {MODEL_PATH}...")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please train the model first using DiffusionModelTrainer.py")
        return

    # Initialize Model
    model = DiffusionModel(INPUT_DIM, HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # Load Normalization Stats
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, "r") as f:
            stats = json.load(f)
        mean = torch.tensor(stats["mean"]).to(DEVICE)
        std = torch.tensor(stats["std"]).to(DEVICE)
        print("Normalization stats loaded.")
    else:
        print(f"Error: Stats file not found at {STATS_PATH}")
        return

    # Generate
    generated_data_norm = sample(model, num_samples)

    # Denormalize
    # x = x_norm * std + mean
    generated_data = generated_data_norm * (std + 1e-8) + mean

    # Calculate 6th dimension: 50 - sum(first 5 dims)
    # This was trained as 5D model with 6th dim inferred
    dim6 = 50.0 - torch.sum(generated_data, dim=1, keepdim=True)
    generated_data = torch.cat([generated_data, dim6], dim=1)

    # Save to JSON
    generated_list = generated_data.cpu().numpy().tolist()

    # Round floats to 2 decimal places
    rounded_list = [[round(float(v), 2) for v in row] for row in generated_list]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(rounded_list, f, indent=4)

    print(f"Successfully generated {num_samples} samples and saved to {output_file}")


if __name__ == "__main__":
    generate_samples()
