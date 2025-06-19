import os
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Dataset Loader with Normalization ---
def load_parquet_scenarios(data_root, batch_size=64, past_len=50, future_len=60):
    all_parquet_paths = glob.glob(os.path.join(data_root, '*', 'scenario_*.parquet'))
    past_traj_list = []
    future_traj_list = []

    for path in all_parquet_paths:
        df = pd.read_parquet(path)
        if 'track_id' in df.columns and 'focal_track_id' in df.attrs:
            agent_id = df.attrs['focal_track_id']
            agent_df = df[df['track_id'] == agent_id]
        else:
            agent_id = df['track_id'].unique()[0]
            agent_df = df[df['track_id'] == agent_id]

        agent_df = agent_df.sort_values('timestep')
        if len(agent_df) < past_len + future_len:
            continue

        past = agent_df.iloc[:past_len][['position_x', 'position_y']].values
        future = agent_df.iloc[past_len:past_len + future_len][['position_x', 'position_y']].values

        # Normalize with respect to last point of past
        anchor = past[-1]
        past -= anchor
        future -= anchor

        if past.shape[0] == past_len and future.shape[0] == future_len:
            past_traj_list.append(torch.tensor(past, dtype=torch.float32))
            future_traj_list.append(torch.tensor(future, dtype=torch.float32))

        if len(past_traj_list) >= batch_size:
            break

    if not past_traj_list:
        raise ValueError("No valid trajectories found in dataset.")

    past_batch = torch.stack(past_traj_list)
    future_batch = torch.stack(future_traj_list)
    noise = torch.randn_like(future_batch) * 5
    negative_batch = future_batch + noise

    return past_batch, future_batch, negative_batch

# --- Model Definitions ---
class ContextEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)

class EnergyModel(nn.Module):
    def __init__(self, traj_len=60, context_size=64, hidden_size=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(traj_len * 2 + context_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, traj, context):
        traj_flat = traj.view(traj.size(0), -1)
        x = torch.cat([traj_flat, context], dim=1)
        return self.mlp(x)

# --- Training Loop ---
def train_ebm(data_path, epochs=100, batch_size=64, lr=0.0005):
    encoder = ContextEncoder()
    energy_model = EnergyModel()
    optimizer = optim.Adam(list(encoder.parameters()) + list(energy_model.parameters()), lr=lr)
    loss_history = []

    print("Starting EBM Training...")
    for epoch in range(epochs):
        past, future, neg = load_parquet_scenarios(data_path, batch_size=batch_size)
        context = encoder(past)
        energy_pos = energy_model(future, context)
        energy_neg = energy_model(neg, context)

        e_pos = -energy_pos
        e_neg = -energy_neg

        loss = -torch.mean(e_pos - torch.logsumexp(torch.cat([e_pos, e_neg]), dim=0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    print("Training Complete.\n")
    return encoder, energy_model, loss_history

# --- Inference: Langevin Sampling ---
def langevin_sampling(encoder, energy_model, data_path, steps=100, alpha=0.1, samples=15):
    past, gt_future, _ = load_parquet_scenarios(data_path, batch_size=1)
    context = encoder(past).detach()

    # Start sampling near ground truth
    init = gt_future.repeat(samples, 1, 1) + torch.randn(samples, 60, 2)
    pred_futures = init.clone().requires_grad_(True)

    for _ in range(steps):
        if pred_futures.grad is not None:
            pred_futures.grad.zero_()

        energy = energy_model(pred_futures, context.repeat(samples, 1)).sum()
        energy.backward()

        with torch.no_grad():
            pred_futures -= 0.5 * alpha * pred_futures.grad
            pred_futures += torch.sqrt(torch.tensor(alpha)) * torch.randn_like(pred_futures)

        pred_futures.requires_grad_(True)

    return past[0].numpy(), gt_future[0].numpy(), pred_futures.detach().numpy()

# --- Visualization ---

def plot_endpoint_scatter(ground_truth, predicted_futures):
    gt_endpoint = ground_truth[-1]
    pred_endpoints = predicted_futures[:, -1, :]

    # Scatter Plot with Ground Truth
    plt.figure(figsize=(6, 6))
    plt.scatter(pred_endpoints[:, 0], pred_endpoints[:, 1], color='red', label='Predicted Endpoints')
    plt.scatter(gt_endpoint[0], gt_endpoint[1], color='green', label='Ground Truth Endpoint')
    plt.title("Predicted vs Ground Truth Endpoints")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # 1. Histogram of Deviations
    deviations = np.linalg.norm(pred_endpoints - gt_endpoint, axis=1)
    plt.figure(figsize=(6, 4))
    plt.hist(deviations, bins=10, color='skyblue', edgecolor='black')
    plt.title("Distribution of Final Point Deviations")
    plt.xlabel("L2 Distance from Ground Truth")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # 2. Hexbin Heatmap of Endpoints
    plt.figure(figsize=(6, 5))
    plt.hexbin(pred_endpoints[:, 0], pred_endpoints[:, 1], gridsize=20, cmap='YlOrRd', edgecolors='grey')
    plt.colorbar(label='Count')
    plt.scatter(gt_endpoint[0], gt_endpoint[1], color='green', s=100, label='Ground Truth', marker='X')
    plt.title("Hexbin Density of Predicted Endpoints")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# --- Main ---
if __name__ == "__main__":
    data_path = "train_data/train/"
    encoder, energy_model, loss_history = train_ebm(data_path, epochs=100)

    # Loss plot
    plt.plot(loss_history, color='purple')
    plt.title("Training Loss (InfoNCE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # Sampling and visualization
    past, ground_truth, pred_samples = langevin_sampling(encoder, energy_model, data_path)
    plot_endpoint_scatter(ground_truth, pred_samples)

    # Summary
    print("\n--- Model Output Summary ---")
    print("Status                 | Value")
    print("------------------------|----------------")
    print(f"Final Training Loss    | {loss_history[-1]:.4f}")
    print(f"Total Epochs           | {len(loss_history)}")
    print(f"Generated Samples      | {pred_samples.shape[0]}")
    print(f"Langevin Steps         | 100")
    print(f"Step Size (alpha)      | 0.1")
