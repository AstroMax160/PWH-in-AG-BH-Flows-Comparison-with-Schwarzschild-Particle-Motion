## Introduction to "Final Parameter Runs" folder

The final model for the standalone PWH (present in 03_Codes/PWH Final Versions), was validated and confirmed to work through a series of tests present in this folder. This file explains the tests and simulations done to validate the model and parameters.

The ParametersSimulation.html file shows a list of parameters that have been shown to ensure walking behaviour, and the walking speed reached in those is approximately *10 mm/s*. Those are values from a different model of simulations, but they have also been observed in the lab. While my code may not be able to replicate everything exactly since the model is different, the walking speeds should still tend to be around 10 mm/s to consider my model as working.

The runs in this folder use the physical parameters from the .html file, but not all the values in the table are present in my model, such as the listed impact drag coefficient, horizontal drag, drag rate and air density. In my model, drag is calculated through the drag eqautionn for a spherical object, such as is done in the Milewski et al PDE model paper. All other physical parameters were replicated exactly in my model. Whenever it is not clarified, the run has 256x256 grid points, and a domain of length 8e-2x8e-2. The "bigdomain" runs are runs with the same domain size as the html file 

## Results discussion

The main key points to evaluate for a run are the spatial resolution dx (determined by number of grid points and domain size) and the steady-state walking speed. For reference in regards to the resolution, the faraday wavelength is 0.00475151 m

**Initial Simulation:** PWH_CPU_u_final_long_256 shows a small domain, small resolution simulation. First simulation with parameters from table and gives a good walking speed (*9.5 mm/s*), with *dx = 3.125e-4 m*

**Isotropic Behaviour and Independence on Initial Velocity:** The walker behaviour should be isotropic (equal in all direction) when there is no background flow (which is the case for these simulations), and independent of the initial droplet velocity (magnitude is independnent). So runs were made using different initial speeds and different directions of it, while keeping everything else the same. The files with label "Diag" and "Down" are the ones where this is shown, and indeed the final walking speed remained very close *9.5 mm/s*, just like in the inital case.  

**Effect of Resolution and Domain Size:** Resolution can affect the simulation results, giving rise to different behaviours and different speeds. So, tests were also performed for the effect of resolution,  which is expected to change walking speed. Furthermore, the effect of the domain size is also analysed, although it is expected that the domain affects results due to the change in resolution. 

- To first test the effect of resolution, the domain was kept the same, but N was increased to 512 points. These tests correspond to files marked with "512" and "512(test2)". For these runs, *dx = 1.562e-4 m*. The resulting walking speed for these simulations was *11.2 mm/s*, an increase from the 9.5, and further from 10 in the parameter file.

- Then, N was increased to 1024, leading to a *dx = 7.812e-5 m*, which caused the walking speed to increase to *11.7 mm/s*

- Then, the domain was increased to a domain similar to the reference simulation. The runs with the bigger domains are labelled 'bigdomain'. The first test was with N=512, leading to *dx = 4.455e-4 m*, which is the biggest dx for any of the simulations (i.e., the worst resolution). The walking speed for this case resulted in *7 mm/s* which is horrible.

- Then, the big domain run was done with N=1024, so, with *dx = 2.227e-4 m*. This is slightly smaller than the base case, and lead to a walking speed of *10.6mm/s*.

So based on this last analysis it seems like the dependency of the walking speed on the resolution is very sensitive for resolutions lower than *dx_max = 3.125e-4 m*, as the walking speed drastically changes below that value. However, increasing the resolution only marginally affects the walking speed by 1 or 2 mm/s. So, I think that, to match the reference values, the aimed for dx should be around *dx_ideal = 2.5e-4 m*.