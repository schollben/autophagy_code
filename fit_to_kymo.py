import os
import code
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import tifffile
import roifile

# Define a 1D Gaussian function
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Define a model that is a sum of multiple Gaussians
def sum_of_gaussians(x, *params):
    num_gaussians = len(params) // 3  # 3 params per Gaussian (A, mu, sigma)
    y = np.zeros_like(x)
    for i in range(num_gaussians):
        A, mu, sigma = params[3*i:3*i+3]
        y += gaussian(x, A, mu, sigma)
    return y

def difference_of_gaussians_model(x, A1, mu1, sigma1, A2, mu2, sigma2):
    """
    Difference of gaussians model for a single particle

    Parameters:
        x (nnumpy.ndarray): x values (pixel positions)
        A1 (float): Amplitude of the positive Gaussian (leading edge)
        mu1 (float): Mean position of the positive Gaussian.
        sigma1 (float): Width of the positive Gaussian
        A2 (float): Amplitude of the negative Gaussian (trailing edge)
        mu2 (float): Mean position of the negative Gaussian
        sigma2 (float): Width of the negative Gaussian
    Returns:
        numpy.ndarray: the difference of Gaussians at each point x.
    """
    gaussian_pos = A1 * np.exp(-(-x - mu1)**2 / (2 * sigma1**2))
    gaussian_neg = A2 * np.exp(-(-x - mu2)**2 / (2 * sigma2**2))
    return gaussian_pos - gaussian_neg

def sum_of_difference_of_gaussians(x, *params):
    """
    Sum of multiple DoGs to model multiple particles

    Parameters:
        x (numpy.ndarray): The x values (pixel positions).
        params: A variable-length argument list, where each group of values
            (A1, mu1, sigma1, A2, mu2, sigma2) represents a DoG for one particle

    Returns:
        numpy.ndarray: The sum of DoGs at each point x.
    """
    y = np.zeros((x.shape))
    for i in range(0, len(params), 4):
        A1, mu1, sigma1, A2, mu2, sigma2 = params[i:i+6]
        y += difference_of_gaussians_model(x, A1, mu1, sigma1, A2, mu2, sigma2)
    return y

def track_particles_with_sum_of_dogs(kymograph, initial_conditions, start_rows, max_mu_shift=5, boundary_pixels=5, max_mu_distance = 10):
    """
    Track particles in a kymograph using a sum of Difference of Gaussians

    Parameters:
        kymograph (numpy.ndarray): 2D array (frames, pixels) representing processed kymograph
        initial_conditions (list): Initial conditions [(A1, mu1, sigma1, A1, mu2, sigma2)] for each particle.
        start_rows (list): Starting row (frame index) for each particle to start tracking.
        max_mu_shift (float): Maximum number of pixels that 'mu1' or 'mu2' (position) that can shift per frame.
        boundary_pixels (int): Number of pixels from the side of the kymo to stop tracking.

    Return:
        tracking_parameters (numpy.ndarray): 2D array of tracked particle parameters
    """
    num_frames, num_cols = kymograph.shape
    num_particles = len(initial_conditions)

    tracked_parameters = np.full((num_frames, 6 * num_particles), np.nan)
    current_params = np.array(initial_conditions).copy().flatten()
    is_tracking_active = [True] * num_particles

    for frame in range(num_frames):
        for i in range(num_particles):
            if frame < start_rows[i]:
                tracked_parameters[frame, 6*i:6*i+6] = current_params[6*i:6*i+6]
                continue

            prev_mu1 = current_params[6 * i + 1]
            prev_mu2 = current_params[6 * i + 4]

            if is_tracking_active[i] and (
                prev_mu1 < boundary_pixels or prev_mu2 < boundary_pixels or
                prev_mu1 > num_cols - boundary_pixels or prev_mu2 > num_cols - boundary_pixels):
                is_tracking_active[i] = False

            if not is_tracking_active[i]:
                continue
            
            y_data = kymograph[frame, :]

            # Forcing mu1, mu2 to be w/in appropriate distance of one another
            #   as well as the previous row positions.
            lower_bound_mu1 = max(0, prev_mu1 - max_mu_shift)
            upper_bound_mu1 = min(num_cols, prev_mu1 + max_mu_shift)
            lower_bound_mu2 = max(lower_bound_mu1 - max_mu_distance, prev_mu2 - max_mu_shift)
            upper_bound_mu2 = min(upper_bound_mu1 + max_mu_distance, prev_mu2 + max_mu_shift)

            bounds = (
                [0, lower_bound_mu1, 1, 0, lower_bound_mu2, 1] * num_particles,
                [np.inf, upper_bound_mu1, np.inf, np.inf, upper_bound_mu2, np.inf] * num_particles
            )
            #code.interact(local=dict(globals(), **locals()))
            initial_guess = current_params

            try:
                params, _ = curve_fit(sum_of_difference_of_gaussians, np.arange(num_cols), y_data, p0=initial_guess, bounds=bounds)
                current_params = params
                tracked_parameters[frame, :] = params

            except RuntimeError:
                tracked_parameters[frame, :] = current_params

    return tracked_parameters

'''
def tracking_difference_of_gaussians(kymograph, initial_conditions, start_rows, max_mu_shift=5, boundary_pixels=5):
    """
    Track particles in a kymograph using the difference of gaussians.
    Meant to be used with a kymograph where np.diff has been applied down rows.
    (think difference kernel)
    
    Parameters:
        kymograph (numpy.ndarray): 2D array (frames x pixels) representing the kymograph.
        initial_conditions (list): Initial conditions [(A, mu, sigma)] for each particle.
        start_rows (list): Starting row (frame index) for each particle.
        max_mu_shift (float): Maximum number of pixels that 'mu' (position) can shift per frame.
        boundary_pixels (int): Number of pixels from the side of the kymograph where tracking should stop.
    
    Returns:
        tracked_parameters (numpy.ndarray): 2D array of tracked parameters (frames x [A1, mu1, sigma1, A2, mu2, sigma2, ...]).
    """
    num_frames, num_cols = kymograph.shape
    num_particles = len(initial_conditions)

    # nan represents zero information
    tracked_particles = np.full((num_frames, 3 * num_particles), np.nan)
    current_params = np.array(initial_conditions).copy()

    is_tracking_active = [True] * num_particles

    for frame in range(num_frames):
        for i in range(num_particles):
            # If the frame is before the particle starts tracking, skip it
            if frame < start_rows[i]:
                tracked_parameters[frame, 3*i:3*i+3] = current_params[i]
                continue
# Stop tracking particle if it enters side boundarys from outside
            prev_mu = current_params[i][1]
            if is_tracking_active[i] and prev_mu < boundary_pixels or prev_mu > num_cols - boundary_pixels:
                is_tracking_active[i] = False # stop tracking
            
            if not is_tracking_active[i]:
                continue # Tracking parameters remain nan

            # Extract the row (frame) of data from the kymograph
            y_data = kymograph[frame, :]

            # Get the previous mu for bounds (use the previous frame's value if available)
            prev_mu = current_params[i][1]

            # Set bounds for mu to restrict its movement within +/- max_mu_shift pixels
            lower_bound = max(0, prev_mu - max_mu_shift)
            upper_bound = min(num_cols, prev_mu + max_mu_shift)

            # Set the bounds for A, mu, and sigma
            bounds = (
                [0, lower_bound, 1],  # Lower bounds for A, mu, and sigma
                [np.inf, upper_bound, np.inf],  # Upper bounds for A, mu, and sigma
            )

            # Initial guess for the fit
            initial_guess = current_params[i]

            # Perform curve fitting with bounds
            try:
                params, _ = curve_fit(gaussian, np.arange(num_cols), y_data, p0=initial_guess, bounds=bounds)

                # Update current parameters for the next iteration
                current_params[i] = params

                # Store the tracked parameters (A, mu, sigma)
                tracked_parameters[frame, 3 * i:3 * i + 3] = params

            except RuntimeError:
                # If the fit fails, keep the previous parameters
                tracked_parameters[frame, 3 * i:3 * i + 3] = current_params[i]

    return tracked_parameters
'''             

def track_particles_with_bounds(kymograph, initial_conditions, start_rows, max_mu_shift=5, boundary_pixels=3, brightness_threshold = 0.5, bright_thresh_duration=5):
    """
    Track particles in a kymograph using curve fitting, with bounds on how much mu can move per frame.
    
    Parameters:
        kymograph (numpy.ndarray): 2D array (frames x pixels) representing the kymograph.
        initial_conditions (list): Initial conditions [(A, mu, sigma)] for each particle.
        start_rows (list): Starting row (frame index) for each particle.
        max_mu_shift (float): Maximum number of pixels that 'mu' (position) can shift per frame.
        boundary_pixels (int): Number of pixels from the side of the kymograph where tracking should stop.
        brightness_threshold (float): Proportion of peak brightness at which a particle stops being tracked (for fading out)
        bright_thresh_duration (int): Number of consecutive frames below brightness threshhold at which tracking stops.
    Returns:
        tracked_parameters (numpy.ndarray): 2D array of tracked parameters (frames x [A1, mu1, sigma1, A2, mu2, sigma2, ...]).
    """
    num_frames, num_cols = kymograph.shape
    num_particles = len(initial_conditions)
    
    # Initialize an array to store the tracked parameters
    # Shape: (num_frames, 3 * num_particles) to store A, mu, sigma for each particle
    #tracked_parameters = np.zeros((num_frames, 3 * num_particles))
    tracked_parameters = np.full((num_frames, 3 * num_particles), np.nan)

    # Initialize the current parameters for each particle
    current_params = np.array(initial_conditions).copy()
    #code.interact(local=dict(globals(), **locals()))
    peak_brightness = [initial[0] for initial in initial_conditions]

    # What particles have stopped tracking bc boundary conditions?
    is_tracking_active = [True] * num_particles
    below_threshhold_count = [0] * num_particles

    # Loop through each frame of the kymograph
    for frame in range(num_frames):
        for i in range(num_particles):
            # If the frame is before the particle starts tracking, skip it
            if frame < start_rows[i]:
                tracked_parameters[frame, 3*i:3*i+3] = current_params[i]
                continue
            
            # Stop tracking particle if it enters side boundarys from outside
            prev_mu = current_params[i][1]
            if is_tracking_active[i] and prev_mu < boundary_pixels or prev_mu > num_cols - boundary_pixels:
                is_tracking_active[i] = False # stop tracking
            
            if not is_tracking_active[i]:
                continue # Tracking parameters remain nan

            # Extract the row (frame) of data from the kymograph
            y_data = kymograph[frame, :]

            # Get the previous mu for bounds (use the previous frame's value if available)
            prev_mu = current_params[i][1]

            # Set bounds for mu to restrict its movement within +/- max_mu_shift pixels
            lower_bound = max(0, prev_mu - max_mu_shift)
            upper_bound = min(num_cols, prev_mu + max_mu_shift)

            # Set the bounds for A, mu, and sigma
            bounds = (
                [0, lower_bound, 1],  # Lower bounds for A, mu, and sigma
                [np.inf, upper_bound, np.inf],  # Upper bounds for A, mu, and sigma
            )

            # Initial guess for the fit
            initial_guess = current_params[i]

            # Perform curve fitting with bounds
            try:
                params, _ = curve_fit(gaussian, np.arange(num_cols), y_data, p0=initial_guess, bounds=bounds)

                # Check if brightness has dimmed sufficiently to stop tracking
                if params[0] < brightness_threshold * peak_brightness[i]:
                    below_threshhold_count[i] += 1
                    if below_threshhold_count[i] >= bright_thresh_duration:
                        is_tracking_active[i] = False
                        continue
                else:
                    below_threshhold_count[i] = 0 # Bright fit, reset counter

                # Update current parameters for the next iteration
                current_params[i] = params

                # Store the tracked parameters (A, mu, sigma)
                tracked_parameters[frame, 3 * i:3 * i + 3] = params

            except RuntimeError:
                # If the fit fails, keep the previous parameters
                tracked_parameters[frame, 3 * i:3 * i + 3] = current_params[i]

    return tracked_parameters

def track_nonmoving_particles(min_prominence, min_width, min_height, verbose=False):
    """
    Subtract Ch1 information from Ch2 kymograph. Peaks of sufficient size
    and width are considered nonmoving autophagosomes.

    Parameters:
        min_prominence (float): Lower bound for peak prominence relative to surround.
        min_width (float): Lower bound for peak width to accept.
        min_height (float): Lower bound for peak height to accept.
        verbose (bool): True to show plots.
    Returns:
        fit_data (np.array): Row with summary statistic values for this particle.
    """
    single_roifile = glob('*.roi')
    if 'RoiSet.zip' in os.listdir():
            roi_list = roifile.roiread('RoiSet.zip')
    elif any(single_roifile):
            roi_list = roifile.roiread(single_roifile[0])
            roi_list = [roi_list]

    results = []

    for i in range(len(roi_list)):
        roi_coords = roi_list[i].coordinates()
        dx = np.diff(roi_coords[:,1])
        dy = np.diff(roi_coords[:,0])
        dendrite_roi_length = np.hypot(dx, dy).sum()
        ch1_kymo = tifffile.imread(os.path.join('kymographs',f'ch1_kymo_roi_{i}.tif'))
        ch2_kymo = tifffile.imread(os.path.join('kymographs',f'ch2_kymo_roi_{i}.tif'))
        ch1_avg_row = np.mean(ch1_kymo, axis=0)
        ch2_avg_row = np.mean(ch2_kymo, axis=0)
        ch2_sub_ch1_row = np.subtract(ch2_avg_row, ch1_avg_row)

        peaks, _ = find_peaks(ch2_sub_ch1_row, prominence=min_prominence, width=min_width, height=min_height)
        #peaks, _ = find_peaks(ch2_sub_ch1_row)
        peakvals = ch2_sub_ch1_row[peaks]
        
        fig, axs = plt.subplots(1,1)
        axs.plot(ch2_sub_ch1_row)
        axs.plot(peaks, peakvals, "x")
        fig.suptitle('Avg Ch2 row subtracted by avg Ch1 row, with peaks')
        fig.savefig(os.path.join('kymographs', f'nonmoving_tracking_roi_{i}_prom-{min_prominence}_wid-{min_width}.png'))
        if verbose:
            fig.show()
        
        for particle in range(peaks.shape[0]):
            # ROInum, dend_mask_len(px),prop_moving, prop_still, avg_velocity, total_displacement, num_direction_changes
            results.append([i, dendrite_roi_length, 0, 1, 0, 0, 0])
        
    return results
    