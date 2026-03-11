import torch
import torch.nn as nn
import os

# Must match scalpel model exactly
class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

domains = ["History", "Science", "Physics", "Chemistry", "Biology"]

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Load outer model
outer_model = SimpleMLP(input_size=32, output_size=len(domains))
outer_model.load_state_dict(torch.load("checkpoints/outer_model.pt"))
torch.save(outer_model, os.path.join(output_dir, "Outer_Scalpel.melm"))

# Load expert models
for domain in domains:
    os.makedirs(os.path.join(output_dir, domain), exist_ok=True)
    expert_model = SimpleMLP(input_size=32, output_size=32)  # must match expert MLP
    expert_model.load_state_dict(torch.load(f"checkpoints/{domain}_expert.pt"))
    torch.save(expert_model, os.path.join(output_dir, domain, f"{domain}.melm"))

print("✅ Export complete. Models saved in output/ folder.")