import os
from glob import glob

import numpy as np
import tifffile
import roifile

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params

def initial_registration(work_dir, ch1_threshold):
    ''' Motion correction of Ch2 data, using Ch1 data.
    - Frames of Ch1 stack over threshold are removed.
    - Rigid, then nonrigid motion correction are done on ch1.
    - Template created from motion corrected ch1.
    - Ch1 registration template applied to the contents of Ch2 for rigid, then nonrigid.
    Check movies in ImageJ/FIJI to determine appropriate size for threshold.
    Performance may be improved with filtering on Ch1 to get rid of horizontal streaks instead.
    '''
    os.chdir(work_dir)
    # Setting parameters for CaImAn's motion correction
    fr = 30           
    decay_time = 1  
    max_shifts = (20, 20)
    strides = (10, 10)
    overlaps = (10, 10)
    max_deviation_rigid = 20

    mc_dict = {
        'fr': fr,
        'decay_time': decay_time,
        'pw_rigid': False,
        'max_shifts': max_shifts,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': 'copy',
        'nonneg_movie': False,
        'use_cuda': False,
        'niter_rig': 5
    }

    mc_dict['upsample_factor_grid'] = 8 # Attempting to fix subpixel registration issue

    # Removing results of previous run, if present
    remove = glob('ch[1,2]*.tif')
    for f in remove:
        os.remove(f)

    fnames_ch1 = glob('*Ch1*')
    fnames_ch2 = glob('*Ch2*')

    raw_ch1_mov = tifffile.imread(fnames_ch1[0])
    raw_ch2_mov = tifffile.imread(fnames_ch2[0])


    # Identify frames with any pixel above the ch1 threshold
    suprathresh_ch1_frames = np.any(raw_ch1_mov > ch1_threshold, axis=(1, 2))
    # Invert to get frames below threshold
    below_thresh_frames = ~suprathresh_ch1_frames
    # Extract clean stack
    ch1_clean_stack = raw_ch1_mov[below_thresh_frames]
    # Handle edge cases
    if ch1_clean_stack.size == 0:
        print(f"Warning: no frames below threshold ({ch1_threshold}). Using entire stack instead.")
        ch1_clean_stack = raw_ch1_mov.copy()
    elif ch1_clean_stack.shape[0] < 10:
        print(f"Warning: only {ch1_clean_stack.shape[0]} frame(s) below threshold. Using full stack for robustness.")
        ch1_clean_stack = raw_ch1_mov.copy()
    # Compute clean template
    ch1_clean_template = np.mean(ch1_clean_stack, axis=0)
    
    
    
    tifffile.imwrite('ch1_subthreshhold_stack.tif', ch1_clean_stack)
    tifffile.imwrite('ch1_stack_template.tif', ch1_clean_template)

    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    
    # Rigid run of Ch1
    mc_dict['fnames'] = ['ch1_subthreshhold_stack.tif']
    opts = params.CNMFParams(params_dict=mc_dict)
    mc = MotionCorrect(mc_dict['fnames'], dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True, template=ch1_clean_template)

    # Getting results of rigid Ch1
    ch1_clean_rigid_reg = cm.load(mc.mmap_file)
    tifffile.imwrite('ch1_rigid_registered.tif', ch1_clean_rigid_reg)

    ch1_clean_rigid_template = np.mean(ch1_clean_rigid_reg, axis=0)
    
    # Nonrigid run of Ch1
    mc_dict['pw_rigid'] = True
    mc_dict['fnames'] = ['ch1_rigid_registered.tif']
    opts = params.CNMFParams(params_dict=mc_dict)

    mc = MotionCorrect(mc_dict['fnames'], dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True, template=ch1_clean_rigid_template)

    ch1_clean_nonrig_reg = cm.load(mc.mmap_file)
    # This will be the template for Ch2 registration
    ch1_clean_nonrig_reg_template = np.mean(ch1_clean_nonrig_reg, axis=0)
    tifffile.imwrite('ch1_nonrigid_registered.tif', ch1_clean_nonrig_reg)

    # Rigid run of Ch2, then nonrigid run of ch2 both using ch1 template
    mc_dict['pw_rigid'] = False
    mc_dict['fnames'] = fnames_ch2
    opts = params.CNMFParams(params_dict=mc_dict)
    mc = MotionCorrect(mc_dict['fnames'], dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True, template=ch1_clean_rigid_template)

    ch2_rigid_reg = cm.load(mc.mmap_file)
    tifffile.imwrite('ch2_rigid_registered.tif', ch2_rigid_reg)

    mc_dict['pw_rigid'] = True
    mc_dict['fnames'] = ['ch2_rigid_registered.tif']
    opts = params.CNMFParams(params_dict=mc_dict)
    mc = MotionCorrect(mc_dict['fnames'], dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True, template=ch1_clean_rigid_template)

    ch2_nonrig_reg = cm.load(mc.mmap_file)
    tifffile.imwrite('ch2_nonrigid_registered.tif', ch2_nonrig_reg)