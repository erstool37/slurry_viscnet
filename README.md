# 🧪 Slurry Viscnet

**Slurry Viscnet** is a computer vision-based viscometer designed for industrial and laboratory use, specifically targeting **Newtonian fluids**. It leverages deep learning to estimate viscosity directly from video data using hybrid modeling and CFD-trained architectures.

---

## 📁 Repository Structure Before Training

Make sure your dataset is organized as follows:
slurry_viscnet/
├── dataset/
│   └── CFDfluid/
│       └── {projectname}/
│           ├── parameters/
│           └── videos/
├── configs/
│   └── config.yaml


You can modify `configs/config.yaml` to customize training behavior.

---

## 🚀 How to Start Training

```bash
# Step into the project directory
cd slurry_viscnet

# Launch training
bash scripts/dev.sh
