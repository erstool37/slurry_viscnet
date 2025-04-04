# Slurry Viscnet
Slurry Viscnet is a Computer Vision based Viscometer for industrial/laboratory application, targeting Newtonian fluids.

# How repo should look before training
The dataset must be stored in 
1. slurry_viscnet/dataset/CFDfluid/{projectname}/parameters 
2. slurry_viscnet/dataset/CFDfluid/{projectname}/videos

Additional repo changes can be made in configs/config.yaml

# How to start training
1. cd slurry_viscnet
2. bash scripts/dev.sh
3. Visit wandb and enjoy the ride
4. Add spices on configs/config.yaml to fit your taste
