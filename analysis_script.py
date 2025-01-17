"""
Script: tal_autophagosome_tracking.py
Author: Gregory Bond
Date: 2024-11-03
Desc: Tracks autophagosome movement from sparse 2p movies.
        Handles registration between channels, kymograph creation, 
        and tracking of particles w/in the kymograph

Example usage:
    - ImageJ, Ch1 stacks for experiments - get Ch1 threshholds for template
    - do_registration=True, run registration for these days.
    - ImageJ, confirm registration quality + draw polyline ROIs
"""
import os
import code
import glob
from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tifffile
import roifile

from autophagy_code.registration import *
from autophagy_code.create_kymograph import *
from autophagy_code.fit_to_kymo import *
from autophagy_code.sanity_plots import *
from autophagy_code.visualization import *
from autophagy_code.signal_analysis import *

if __name__ == '__main__':
    # settings
    
    #work_dirs = ['/home/gnb/Documents/autophagy_tal/data/Dendrite_Tsc2KO/Dendrites_Tsc2KO_FOV_015',
    #            '/home/gnb/Documents/autophagy_tal/data/Dendrite_Tsc2KO/Dendrites_Tsc2KO_FOV_022',
    #            '/home/gnb/Documents/autophagy_tal/data/Dendrite_Tsc2KO/Dendrites_Tsc2KO_FOV_023',
    #            '/home/gnb/Documents/autophagy_tal/data/Neurite_Tsc2KO/Neurite_Tsc2KO_FOV_007',
    #            '/home/gnb/Documents/autophagy_tal/data/Neurite_Tsc2KO/Neurite_Tsc2KO_FOV_008',
    #            '/home/gnb/Documents/autophagy_tal/data/Neurite_Tsc2KO/Neurite_Tsc2KO_FOV_017']

    work_dirs = ['/home/gnb/Documents/autophagy_tal/data/Axon_Tsc2KO/040',
                 '/home/gnb/Documents/autophagy_tal/data/Axon_Tsc2KO/041',
                 '/home/gnb/Documents/autophagy_tal/data/Axon_WT/01',
                 '/home/gnb/Documents/autophagy_tal/data/Axon_WT/02',
                 '/home/gnb/Documents/autophagy_tal/data/Axon_WT/03',
                 '/home/gnb/Documents/autophagy_tal/data/Axon_WT/04',
                 '/home/gnb/Documents/autophagy_tal/data/Dendrites_Scr/FOV_007',
                 '/home/gnb/Documents/autophagy_tal/data/Dendrites_Scr/FOV_010',
                 '/home/gnb/Documents/autophagy_tal/data/Dendrites_Scr/FOV_017',
                 '/home/gnb/Documents/autophagy_tal/data/Dendrites_Scr/FOV_019',
                 '/home/gnb/Documents/autophagy_tal/data/Dendrites_Scr/FOV_32']
    
    work_dirs = ['/home/gnb/Documents/autophagy_tal/data/Neurite_Tsc2KO/Neurite_Tsc2KO_FOV_007',
                 '/home/gnb/Documents/autophagy_tal/data/Neurite_Tsc2KO/Neurite_Tsc2KO_FOV_008',
                 '/home/gnb/Documents/autophagy_tal/data/Neurite_Tsc2KO/Neurite_Tsc2KO_FOV_017']
    #work_dirs = ['/home/gnb/Documents/autophagy_tal/data/Axon_Scr/Axon_Scr_29']
    # Pixel-distance to sample normal to each PolyLine segment during kymograph creation
    kymo_widths = [7,11,11,7,7,7,7,9,9,7,7,7]
    # 1 - use rigid-registered stacks. 2 - use nonrigid-registered stacks
    kymo_stack_choices = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    
    #work_dirs = ['/home/gnb/Documents/autophagy_tal/data/Dendrites_surface_WT/639B_FOV_003']
    #kymo_widths = [11]

    #kymo_stack_choices = [1]

    do_registration = False
    do_kymos = False        # PolyLine ROIs must be drawn first
    do_tracking = True      # Moving and nonmoving autophagosomes

    #assert len(work_dirs) == len(kymo_widths) == len(kymo_stack_choices), "Mismatch in input lists sizes."

    for it in range(len(work_dirs)):
        wk = work_dirs[it]
        os.chdir(wk)
        path = Path(wk)
        cond_name = path.parent.name # 'Dendrites_surface_WT'
        cond_subname = path.name     # '648B_FOV_004'
        if do_registration:
            print(f"Registering for {wk}")
            # Prompt for ch1_threshold, can open the initial image in imagej each time?
            ch1_thresh = input("Ch1 upper threshold (default is 70):")
            if ch1_thresh == '':
                ch1_thresh = 70
            else:
                ch1_thresh = int(ch1_thresh)
            initial_registration(wk, ch1_thresh)
        
        if do_kymos:
            print(f"Making kymos for: {wk}")
            kymo_width = kymo_widths[it]
            which_kymo = kymo_stack_choices[it]
            if which_kymo == 1:
                ch1 = tifffile.imread('ch1_rigid_registered.tif')
                ch2 = tifffile.imread('ch2_rigid_registered.tif')
            elif which_kymo == 2:
                ch1 = tifffile.imread('ch1_nonrigid_registered.tif')
                ch2 = tifffile.imread('ch2_nonrigid_registered.tif')
            single_roifile = glob('*.roi')
            if 'RoiSet.zip' in os.listdir():
                roi_list = roifile.roiread('RoiSet.zip')
            elif any(single_roifile):
                roi_list = roifile.roiread(single_roifile[0])
                roi_list = [roi_list]
            #code.interact(local=dict(globals(), **locals()))
            for i in range(len(roi_list)):
                roi_coords = roi_list[i].coordinates()
                ch1_kymo, _ = create_kymograph(ch1, roi_coords, kymo_width)
                ch2_kymo, ch2_kymo_movie = create_kymograph(ch2, roi_coords, kymo_width)
                ch2_tracking_kymo = create_tracking_kymo(ch2_kymo, roi_coords)

                if not os.path.exists('kymographs'):
                    os.makedirs('kymographs')
                
                tifffile.imwrite(os.path.join('kymographs',f'ch1_kymo_roi_{i}.tif'), ch1_kymo)
                tifffile.imwrite(os.path.join('kymographs' ,f'ch2_kymo_roi_{i}.tif'), ch2_kymo)
                tifffile.imwrite(os.path.join('kymographs' ,f'ch2_tracking_kymo_roi_{i}.tif'), ch2_tracking_kymo)
                print(ch2_kymo_movie.shape)
                create_kymo_movie(ch2_kymo_movie, f'ch2_kymo_mov_roi_{i}.mp4')

        if do_tracking:
            all_particle_results = []
            # non-moving
            prominence = 1.0 # most important param
            height = 0.0 # Ch2 must be brighter than Ch1
            width = 0.5 # little effect
            
            nonmoving_results = track_nonmoving_particles(prominence, width, height, verbose=True)
            
            for particle in nonmoving_results:
                particle.insert(0,cond_subname)
                particle.insert(0,cond_name)
                all_particle_results.append(particle)

            # moving particles now 
            single_roifile = glob('*.roi')
            if any(single_roifile):
                single_roi = roifile.roiread(single_roifile[0])
                roi_list = [single_roi]
            else:
                roi_list = roifile.roiread('RoiSet.zip')
            for i in range(len(roi_list)):
                roi_coords = roi_list[i].coordinates()
                dx = np.diff(roi_coords[:,1])
                dy = np.diff(roi_coords[:,0])
                dendrite_roi_length = np.hypot(dx, dy).sum()
                ch2_tracking_kymo = tifffile.imread(os.path.join('kymographs', f'ch2_tracking_kymo_roi_{i}.tif'))
                num_frames = ch2_tracking_kymo.shape[0]
                # Try for single ROI
                #code.interact(local=dict(globals(), **locals()))
                trackpoints_filename = glob(f"kymographs/trackpoints_roi_{i}*")
                
                if len(trackpoints_filename) == 0: # No particles here
                    continue
                trackpoints_file = roifile.roiread(trackpoints_filename[0])
                
                if not isinstance(trackpoints_file, list):
                    trackpoints_file = [trackpoints_file]
                
                for j in range(len(trackpoints_file)):
                    particle = trackpoints_file[j]
                    coords = particle.coordinates().astype('int')
                    positions = trackpoints_to_position(coords, num_frames=num_frames)
                    movement_summary = characterize_motion_from_coords(coords=coords)
                    movement_summary.insert(0,dendrite_roi_length)
                    movement_summary.insert(0, i) # which dendrite
                    movement_summary.insert(0,cond_subname)
                    movement_summary.insert(0,cond_name)
                    all_particle_results.append(movement_summary)

        # outside first do_tracking loop
        if do_tracking:
            data_outname = 'particle_summary_data.csv'
            with open(data_outname, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(all_particle_results)
            for row in all_particle_results:
                print(row)
            print(f'Tracking results saved for {cond_name}, {cond_subname}')
            #code.interact(local=dict(globals(), **locals()))
        