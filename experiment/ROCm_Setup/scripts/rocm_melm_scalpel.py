#!/usr/bin/env python3
# rocm_melm_scalpel.py

import torch
import torch.nn as nn
import torch.optim as optim
import os

# ----------------------------
# Dummy dataset: prompts → domain signals
# ----------------------------
prompts = [
    "Explain Newton's laws of motion",
    "Who was Napoleon Bonaparte?",
    "Photosynthesis in plants",
    "Quantum entanglement basics",
    "The French Revolution summary",
    "I need help with my Python application, that needs to be in Spanish language."
]

labels = [
    {"History": 0, "Science": 0, "Physics": 99, "Chemistry": 0, "Biology": 0, "Programming": 0, "Language": 0},
    {"History": 99, "Science": 0, "Physics": 0, "Chemistry": 0, "Biology": 0, "Programming": 0, "Language": 0},
    {"History": 0, "Science": 99, "Physics": 0, "Chemistry": 0, "Biology": 99, "Programming": 0, "Language": 0},
    {"History": 0, "Science": 99, "Physics": 99, "Chemistry": 0, "Biology": 0, "Programming": 0, "Language": 0},
    {"History": 99, "Science": 0, "Physics": 0, "Chemistry": 0, "Biology": 0, "Programming": 0, "Language": 0},
    {"History": 0, "Science": 0, "Physics": 0, "Chemistry": 0, "Biology": 0, "Programming": 99, "Language": 99}
]

domains = list(labels[0].keys())

# ----------------------------
# Simple MLP for demonstration
# ----------------------------
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

# ----------------------------
# Encode prompts into vectors (dummy one-hot / length based)
# ----------------------------
def encode_prompt(prompt):
    vec = torch.zeros(32)
    vec[0: min(len(prompt), 32)] = torch.tensor([ord(c) % 256 for c in prompt[:32]], dtype=torch.float32)
    return vec

X = torch.stack([encode_prompt(p) for p in prompts])
Y = torch.stack([torch.tensor([v for v in lbl.values()], dtype=torch.float32) for lbl in labels])

# ----------------------------
# Outer scalpel model: predicts top domain
# ----------------------------
outer_model = SimpleMLP(input_size=32, output_size=len(domains))
criterion = nn.MSELoss()
optimizer = optim.Adam(outer_model.parameters(), lr=0.01)

# Train outer model
for epoch in range(50):
    optimizer.zero_grad()
    outputs = outer_model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 - Loss: {loss.item():.4f}")

# Determine top domain per prompt
with torch.no_grad():
    preds = outer_model(X)
    for i, p in enumerate(prompts):
        signal = {domain: int(preds[i,j]*100) for j, domain in enumerate(domains)}
        top_domain = max(signal, key=signal.get)
        print(f"\nPrompt: {p}")
        print(f"Predicted Signal: {signal}")
        print(f"Top Module -> {top_domain}")

# ----------------------------
# Save models for export
# ----------------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(outer_model.state_dict(), "checkpoints/outer_model.pt")

# Save a dummy expert per domain
expert_models = {}
for domain in domains:
    model = SimpleMLP(input_size=32, output_size=32)  # expert MLP
    expert_models[domain] = model
    torch.save(model.state_dict(), f"checkpoints/{domain}_expert.pt")

print("\n✅ Training complete. Checkpoints saved to ./checkpoints/")