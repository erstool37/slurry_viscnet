# Fluid Masking Simulation
Based on http://madebyevan.com/webgl-water/, few adjustions have been made
1. Masking function based on reflected light intensity thresholding, fresnel ratio altering, and color altering
2. Realistic dual lighting
3. Capture function two download raw and masked images
4. main.js that drops random droplets on mesh (requires change in index.html)
5. mainVortex.js that stamps random vortices on mesh

# How to Launch
1. Launch Terminal
2. Navigate to /stirsimul
3. Start server > python -m http.server 8000
4. Launch Browser > http://localhost:8000
5. Make sure both pools are visible on browser, and alternations of the canvas size during the process is not allowed.
6. Image Captures are initiated automatically, and will start download after all 960files are stored.


# Errors
1. After pausing the simulation, the left viewport cedes to function for a while