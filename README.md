# 🧪 Slurry Viscnet

**Slurry Viscnet** is a computer vision-based viscometer designed for industrial and laboratory use, specifically targeting **Newtonian fluids**. It leverages deep learning to estimate viscosity directly from video data using CFD generated datasets.

---

## 📁 Repository Structure Before Training

Make sure your dataset is organized as follows:
slurry_viscnet/dataset/{projectname}/parameters
slurry_viscnet/dataset/{projectname}/videos

You can modify `configs/config.yaml` to customize training behavior.

---

## 🚀 How to Start Training

```bash
# Step into the project directory
cd slurry_viscnet

# Launch training
bash scripts/dev.sh
