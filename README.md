![staticvortex](https://github.com/user-attachments/assets/fb14f517-61c2-45de-ba7e-14aff85b2080)# 🧪 Slurry Viscnet

**Slurry Viscnet** is a computer vision-based viscometer designed for industrial and laboratory use, specifically targeting **Newtonian fluids**. It leverages deep learning to estimate viscosity directly from video data using CFD generated datasets.

## 📁 Repository Structure Before Training

Make sure your dataset is organized as follows:
slurry_viscnet/dataset/{projectname}/parameters
slurry_viscnet/dataset/{projectname}/videos

You can modify `configs/config.yaml` to customize training behavior.

### Example dataset

#### Static vortex;
![Static Vortex](datasets/assets/staticvortex.gif)

#### Decaying vortex
![Decaying Vortex](datasets/assets/decayingvortex.gif)

## 🚀 How to Start Training

```bash
cd slurry_viscnet
bash scripts/setup.sh
bash scripts/dev.sh
