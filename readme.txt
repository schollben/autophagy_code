# readme

--------------
Getting started: 

- Brew installation, then: 
    - `brew install git`
    - `brew install miniconda`

- `conda init` or `conda init zsh` if you're using macOS and zshell

- `conda config --add channels conda-forge`

- `conda create -n autophagy python=3.11 caiman`

- `conda activate autophagy`

- `pip install roifile`

- `pip install tifffile`

- `pip install imagecodecs`
 
- Install vs code from Windows

- vs code - install two extensions:
	- Python
	- Python Extensions

- `git pull` repo for the code itself
 
 


--------------
Steps are as follows :
    1.) Figure out threshold value for Ch1 to kick out streak-frames (ImageJ/FIJI)
    2.) Apply threshold to Ch1 images and perform registration (Python)
    3.) Confirm registration quality is usable, then draw PolyLine ROI on cell arbor for analysis (ImageJ/FIJI)
    4.) Create tracking kymograph images for each PolyLine ROI drawn (Python)
    5.) Draw particle trajectories on created kymograph images (ImageJ/FIJI)
    6.) Extract all particle tracking information into a .csv file (Python)

Here is the registration process description.
- The user is required to provide a threshold value. All frames in the ch1 data that have pixel intensities
higher than the threshold are removed from further analysis. This is to remove the effect of the bright
horizontal 'smearing' artefacts that are present in some days.
- The resulting frames from ch1 are then rigid-registered. The template each frame is being registered to is
simply the average of all post-threshold frames in ch1.
- The results of the ch1 rigid registration should have the majority of motion removed. To remove additional motion,
piecewise nonrigid registration is performed. This should remove small movements, but should also
avoid 'shunting' autophagosome locations towards bright structures (nonmoving vesicles, spines, crossings in arbor) 
once they get sufficiently close to each other.

Now that ch1 has been registered correctly, a template image is created from it for ch2 registration.
We do this because ch2 cannot serve as its own template because the vesicles themselves are moving.
A template that incorporates just structural information works better.

- Ch2 is rigid-registered with the ch1 template.
- Ch2 rigid registration then has the piecewise-rigid (nonrigid) registration performed again using the ch1 template.


Here are the files that will be created as a result of running the registration.
- ch1_subthreshold_stack.tif - Ch1 data, but only including frames that are below the provided pixel-intensity threshold.
- ch1_rigid_registered.tif - Post-threshold ch1 data that's been rigid registered.
- ch1_nonrigid_registered.tif - Result of applying piecewise-rigid (nonrigid) registration to the ch1_rigid stack.
- ch1_stack_template.tif - The resulting template image from ch1 that will be used for ch2 registration
- ch2_rigid_registered.tif - Ch2 data rigid registered, using the ch1 average as a template to register to.
- ch2_nonrigid_registered.tif - Ch2 rigid-registered data, now piecewise-rigid (nonrigid) registered to ch1 template image.


To check quality at this stage :
- ch1_subthreshold_stack.tif should have a good number of frames in it. Too low of a threshold (especially if there
are bright structures like soma) will result in almost all (or all) frames being kicked out. Too high of a threshold
will not kick out the horizontal smearing.
- ch1_nonrigid_registered.tif - There should not be movement of more than a few pixels here. If there is tearing
in the image, check the rigid registration to determine if the majority of movement was removed in this first stage.
Piecewise rigid (nonrigid) motion correction does not work well if the rigid motion correction was insufficient.

- ch2_nonrigid_registered.tif - See whether there's any significant motion here, just the same as ch1.
Another situation to look out for is when vesicles move near very bright, nonmoving structures.
If there are nonmoving vesicles that moving vesicles pass near, there can be cases where the registration
'snaps' the position of the moving vesicle to the nonmoving vesicle, then 'snaps' it back to the correct
position once it moves further away. This is because we're using a template-based registration and our templates
only have the structural information.


Once the quality of the registration has been confirmed, the ROIs for each dendrite/axon must be drawn.
These ROIs are used to create the kymograph images that show particle movement over time.
- In ImageJ/FIJI, use the 'segmented line' ROI (right click on the straight line ROI, then select segmented).
- Open ch2_nonrigid_registered.tif
- Left click to start the ROI, as well as place vertices for the segmented line.
- Right click to place the final point. Hit 't' to add this to the ROI Manager.