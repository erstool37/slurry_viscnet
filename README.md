# 🧪 Slurry Viscnet

**Slurry Viscnet** is a computer vision-based viscometer designed for industrial and laboratory use, specifically targeting **Newtonian fluids**. It leverages deep learning to estimate viscosity directly from video data using CFD generated datasets.

---

## 📁 Repository Structure Before Training

Make sure your dataset is organized as follows:
slurry_viscnet/dataset/{projectname}/parameters
slurry_viscnet/dataset/{projectname}/videos

You can modify `configs/config.yaml` to customize training behavior.

---

## Example dataset

# Static vortex; vicosity 10cP
GIFGIF
# Decaying vortex; viscosity 10cP and 1000cP (수치는 바꿔도 될듯!)
GIFGIF


## 🚀 How to Start Training

```bash
cd slurry_viscnet
bash scripts/setup.sh
bash scripts/dev.sh
