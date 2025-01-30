# Fluid Masking Simulation
Based on http://madebyevan.com/webgl-water/

# How to Launch
1. Launch Terminal
2. Navigate to /stirsimul
3. Start server > python -m http.server 8000
4. Launch Browser > http://localhost:8000
5. Make sure both pools are visible on browser, and alternations of the canvas size during the process is not allowed.
6. Image Captures are initiated automatically, and will start download after all 960files are stored.

# Masking Logic
1. Surface+Underwater color altering
2. Refraction ratio(fresnel) altering

# Errors
1. After pausing the simulation, the left viewport cedes to function for a while
2. Reflection of the light source might deteriorate the training efficiency
3. When capturing images, there are sudden jumps in the captures.