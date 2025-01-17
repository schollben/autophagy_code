import os
import code

from glob import glob

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.interpolate import UnivariateSpline

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog


import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params

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

def fit_gaussians_to_kymograph(kymograph, initial_conditions, start_rows, width):
    num_rows, num_cols = kymograph.shape
    num_particles = len(initial_conditions)
    
    # Preallocate results matrix (rows x particles x 3) for (A, mu, sigma)
    fit_results = np.zeros((num_rows, num_particles * 3))
    
    x_data = np.arange(num_cols)
    
    # Initialize the current parameters for each particle
    current_params = initial_conditions

    # Loop through each row of the kymograph
    for row in range(num_rows):
        y_data = kymograph[row, :]
        
        # Set all values below zero to zero
        y_data = np.clip(y_data, 0, None)
        
        # Build a list to hold the initial guess for fitting
        active_params = []
        active_conditions = []
        
        # Determine which particles are active in this row
        for i, start_row in enumerate(start_rows):
            if row >= start_row:
                active_params.extend(current_params[i])  # Use the current params for active particles
                active_conditions.append(i)
        
        if active_params:
            try:
                # Fit the sum of Gaussians model to the current row
                params, _ = curve_fit(sum_of_gaussians, x_data, y_data, p0=active_params)
                
                # Update the current parameters for each active particle
                for i, particle_index in enumerate(active_conditions):
                    current_params[particle_index] = params[3*i:3*i+3]
                    fit_results[row, particle_index*3:(particle_index+1)*3] = current_params[particle_index]
                
            except RuntimeError:
                # If fitting fails, set the active particle columns to np.nan
                for particle_index in active_conditions:
                    fit_results[row, particle_index*3:(particle_index+1)*3] = np.nan
        # If no particles are active, leave the row as zeros (already initialized to zero
    return fit_results


def track_particles_with_bounds(kymograph, initial_conditions, start_rows, max_mu_shift=5):
    """
    Track particles in a kymograph using curve fitting, with bounds on how much mu can move per frame.
    
    Parameters:
        kymograph (numpy.ndarray): 2D array (frames x pixels) representing the kymograph.
        initial_conditions (list): Initial conditions [(A, mu, sigma)] for each particle.
        start_rows (list): Starting row (frame index) for each particle.
        max_mu_shift (float): Maximum number of pixels that 'mu' (position) can shift per frame.
    
    Returns:
        tracked_parameters (numpy.ndarray): 2D array of tracked parameters (frames x [A1, mu1, sigma1, A2, mu2, sigma2, ...]).
    """
    num_frames, num_cols = kymograph.shape
    num_particles = len(initial_conditions)
    
    # Initialize an array to store the tracked parameters
    # Shape: (num_frames, 3 * num_particles) to store A, mu, sigma for each particle
    tracked_parameters = np.zeros((num_frames, 3 * num_particles))
    
    # Initialize the current parameters for each particle
    current_params = np.array(initial_conditions).copy()

    # Loop through each frame of the kymograph
    for frame in range(num_frames):
        for i in range(num_particles):
            # If the frame is before the particle starts tracking, skip it
            if frame < start_rows[i]:
                tracked_parameters[frame, 3*i:3*i+3] = current_params[i]
                continue
            
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

def visualize_fit_results(kymograph, fit_results, start_rows, num_particles):
    """
    Visualize the fit results by plotting the kymograph and marking the mu positions
    of the Gaussian fits for each particle with small circles.
    
    Parameters:
        kymograph (numpy.ndarray): 2D array of the kymograph image (rows=time, cols=space).
        fit_results (numpy.ndarray): 2D array of the fit results (rows=time, cols=3*num_particles).
        start_rows (list of int): List of row indices where tracking of each particle begins.
        num_particles (int): The number of particles being tracked.
    """
    plt.figure(figsize=(10, 6))
    
    # Display the kymograph image
    plt.imshow(kymograph, aspect='auto', cmap='gray', origin='lower')
    
    # Define a set of colors for each particle
    colors = plt.cm.get_cmap('tab10', num_particles)  # Use a colormap for distinct colors

    # Loop through each particle
    for particle_index in range(num_particles):
        # Extract the mu (mean position) for this particle
        mu_values = fit_results[:, 3*particle_index + 1]  # The second parameter is mu (mean)
        
        # Create a mask for rows where the particle is being tracked
        active_mask = np.arange(fit_results.shape[0]) >= start_rows[particle_index]
        
        # Plot the mu values with circles for the active tracking rows
        plt.plot(mu_values[active_mask], np.arange(fit_results.shape[0])[active_mask], 
                 'o', color=colors(particle_index), label=f"Particle {particle_index+1}")
    
    plt.xlabel('Position (space)')
    plt.ylabel('Time (rows)')
    plt.colorbar(label='Kymograph Intensity')
    plt.title('Particle Tracking with Gaussian Fit Results')
    plt.legend()
    plt.show()

def create_kymograph(stack, polyline_roi, width):
    """
    Create a kymograph from a 3D image stack (time, width, height) using a polyline ROI.
    Averaging is done over a width normal to the polyline.

    Parameters:
        stack (numpy.ndarray): 3D array (time, width, height) of calcium movie.
        polyline_roi (numpy.ndarray): 2D array (n_points, 2) of (y, x) coordinates defining the polyline.
        width (int): Width of the normal lines to average across (in pixels).

    Returns:
        kymograph (numpy.ndarray): Kymograph as a 2D array (time, polyline length).
    """
    # Get the number of time points
    num_frames = stack.shape[0]
    
    # Swap the (x, y) order in polyline_roi (now assuming polyline_roi is (y, x))
    polyline_roi_swapped = np.fliplr(polyline_roi)  # Flipping (y, x) to (x, y)

    # Calculate distances between consecutive polyline points
    distances = np.sqrt(np.sum(np.diff(polyline_roi_swapped, axis=0)**2, axis=1))
    total_distance = int(np.sum(distances))
    
    # Calculate unit vectors along each polyline segment
    unit_vectors = np.diff(polyline_roi_swapped, axis=0)
    unit_vectors = unit_vectors / np.linalg.norm(unit_vectors, axis=1)[:, None]
    
    # Preallocate kymograph array (time, distance along polyline)
    kymograph = np.zeros((num_frames, total_distance))
    
    # Keep track of the current position along the kymograph
    current_position = 0
    
    # Loop through each polyline segment
    for i, (start, end) in enumerate(zip(polyline_roi_swapped[:-1], polyline_roi_swapped[1:])):
        segment_length = int(distances[i])
        
        # Normal vector perpendicular to the polyline segment
        normal_vector = np.array([-unit_vectors[i, 1], unit_vectors[i, 0]])
        
        # Sample along the segment
        for j in range(segment_length):
            alpha = j / segment_length
            point = (1 - alpha) * start + alpha * end  # Interpolate along the segment
            
            # For each time point, extract and average pixel values normal to the segment
            for t in range(num_frames):
                # Sample points along the normal direction to the segment
                normal_points = np.array([point + k * normal_vector for k in range(-width // 2, width // 2)])
                
                # Get the pixel values using map_coordinates (interpolates at sub-pixel locations)
                intensities = map_coordinates(stack[t], normal_points.T, order=1)
                
                # Average the intensities and store in the kymograph
                avg_intensity = np.mean(intensities)
                kymograph[t, current_position + j] = avg_intensity
        
        current_position += segment_length
    
    return kymograph

def initial_registration(work_dir, ch1_threshold, mc_dict):
    ''' Motion correction of Ch2 data, using Ch1 data.
    - Frames of Ch1 stack over threshold are removed.
    - Rigid, then nonrigid motion correction are done on ch1.
    - Template created from motion corrected ch1.
    - Ch1 registration template applied to the contents of Ch2 for rigid, then nonrigid.

    Check movies in ImageJ/FIJI to determine appropriate size for threshold.

    Performance may be improved with filtering on Ch1 to get rid of horizontal streaks instead.
    '''
    os.chdir(work_dir)

    # Removing results of previous run, if present
    remove = glob('ch[1,2]*.tif')
    for f in remove:
        os.remove(f)

    fnames_ch1 = glob('*Ch1*')
    fnames_ch2 = glob('*Ch2*')

    raw_ch1_mov = tifffile.imread(fnames_ch1[0])
    raw_ch2_mov = tifffile.imread(fnames_ch2[0])

    suprathresh_ch1_frames = np.any(raw_ch1_mov > ch1_threshold, axis=(1,2))
    ch1_clean_stack = raw_ch1_mov[np.logical_not(suprathresh_ch1_frames)]
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
    #code.interact(local=dict(globals(), **locals())) 


def select_initial_conditions(kymograph, default_sigma=5):
    """
    Allows the user to click on the kymograph to select starting points for tracking particles.
    Left-click to select points, right-click to deselect the last point.
    The user can click as many points as they want and close the window to finish selection.
    
    Parameters:
        kymograph (numpy.ndarray): 2D array representing the kymograph (rows=time, cols=space).
        default_sigma (float): Initial guess for the width of the Gaussian.
    
    Returns:
        initial_conditions (list): List of initial conditions [(A, mu, sigma), ...] for each particle.
        start_rows (list): List of starting rows for each particle.
    """
    fig, ax = plt.subplots()
    ax.imshow(kymograph, aspect='auto', cmap='gray', origin='lower')
    plt.title('Click on particles to initialize tracking.\nLeft-click to select, right-click to remove last point.\nClose the window when done.')

    # List to store clicked points
    clicks = []

    # Callback function to handle clicks
    def onclick(event):
        if event.button == 1:  # Left-click to add a point
            clicks.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro')  # Mark the clicked point
            plt.draw()
        elif event.button == 3:  # Right-click to remove the last point
            if clicks:
                clicks.pop()
                ax.clear()
                ax.imshow(kymograph, aspect='auto', cmap='gray', origin='lower')
                plt.title('Click on particles to initialize tracking.\nLeft-click to select, right-click to remove last point.\nClose the window when done.')
                # Re-draw the remaining points
                for click in clicks:
                    ax.plot(click[0], click[1], 'ro')
                plt.draw()

    # Connect the event handler to the figure
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    # Show the plot and block execution until the window is closed
    plt.show(block=True)

    # Disconnect the event handler after the plot is closed
    fig.canvas.mpl_disconnect(cid)

    # Process the selected points and return the initial conditions
    initial_conditions = []
    start_rows = []

    for click in clicks:
        mu = click[0]  # The x-coordinate (mu: position within the row)
        start_row = int(click[1])  # The y-coordinate (row number)
        amplitude = kymograph[start_row, int(mu)]  # Amplitude from the pixel value
        initial_conditions.append([amplitude, mu, default_sigma])  # Store amplitude, mu, sigma
        start_rows.append(start_row)
    
    return initial_conditions, start_rows


def filter_and_compute_derivatives(fit_results, frame_time_step, filter_width, start_rows):
    """
    Applies a zero-phase filter (using filtfilt) to the position data (from fit_results),
    and computes the first derivative (velocity) and second derivative (acceleration) using np.diff,
    starting only from the respective `start_row` for each particle.
    
    Parameters:
        fit_results (numpy.ndarray): Array of fit results (rows=frames, cols=3*particles).
                                     Each particle has 3 columns: (A, mu, sigma), where mu is the position.
        frame_time_step (float): The time difference between consecutive frames (e.g., time per frame).
        filter_width (int): The width of the moving average filter to apply using filtfilt.
        start_rows (list): List of start rows (frame indices) for each particle.
        
    Returns:
        filtered_positions (numpy.ndarray): Smoothed positions from the filter.
        velocity (numpy.ndarray): Velocity of each particle over time.
        acceleration (numpy.ndarray): Acceleration of each particle over time.
    """
    num_frames, num_cols = fit_results.shape
    num_particles = num_cols // 3  # Each particle has 3 columns (A, mu, sigma)
    
    # Initialize arrays to store filtered positions, velocity, and acceleration
    filtered_positions = np.zeros((num_frames, num_particles))
    velocity = np.zeros((num_frames, num_particles))  # Initialize velocity with the same size
    acceleration = np.zeros((num_frames, num_particles))  # Initialize acceleration with the same size
    
    # Create the filter coefficients for a simple moving average filter
    b = np.ones(filter_width) / filter_width  # Numerator coefficients (moving average)
    a = 1  # Denominator coefficients (no feedback, FIR filter)
    
    # Loop through each particle and apply the filter and derivatives
    for i in range(num_particles):
        # Extract the 'mu' (position) column for the current particle
        mu_column = fit_results[:, 3 * i + 1]  # 'mu' is at index 1 of each particle's fit

        # Only consider data starting from the respective start_row for each particle
        start_row = start_rows[i]
        valid_mu = mu_column[start_row:]  # Ignore data before start_row
        
        # Apply the zero-phase filter to the valid position data with padding
        filtered_mu = filtfilt(b, a, valid_mu, padlen=filter_width + 10)

        # Compute the velocity (first derivative) using np.diff
        velocity[start_row + 1:, i] = np.diff(filtered_mu) / frame_time_step

        # Compute the acceleration (second derivative) using np.diff on the velocity
        acceleration[start_row + 2:, i] = np.diff(velocity[start_row + 1:, i]) / frame_time_step

        # Store the filtered positions (aligning back with the original indices)
        filtered_positions[start_row:, i] = filtered_mu

    return filtered_positions, velocity, acceleration

def lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    """
    Applies a low-pass Butterworth filter to the input data.
    
    Parameters:
        data (numpy.ndarray): The input data to be filtered (e.g., position data).
        cutoff_freq (float): The cutoff frequency for the filter (in Hz).
        sample_rate (float): The sample rate of the data (inverse of the frame time step, in Hz).
        order (int): The order of the Butterworth filter (higher = sharper cutoff).
    
    Returns:
        filtered_data (numpy.ndarray): The filtered data.
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def plot_position_velocity_acceleration(time, position, velocity, acceleration, start_rows, events):
    """
    Plots position, velocity, and acceleration in three vertically stacked subplots with a shared x-axis,
    with background colors based on the events.

    Parameters:
        time (numpy.ndarray): 1D array of time points corresponding to each frame.
        position (numpy.ndarray): 2D array of position (num_frames, num_particles).
        velocity (numpy.ndarray): 2D array of velocity (num_frames - 1, num_particles).
        acceleration (numpy.ndarray): 2D array of acceleration (num_frames - 2, num_particles).
        start_rows (list): List of start rows (frame indices) for each particle.
        events (list): List of tuples (event, index) representing movement events.
    """
    num_particles = position.shape[1]

    # Create a figure with three subplots, stacked vertically, and share the x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Define colors for each event type
    event_colors = {
        'turnaround': 'lightcoral',
        'stop_and_continue': 'lightblue',
        'stopped': 'lightgray',
        'started': 'lightgreen',
    }

    # Plot background colors based on the events
    for event, index in events:
        color = event_colors.get(event, 'white')
        ax1.axvspan(time[index], time[index+1], color=color, alpha=0.3)
        ax2.axvspan(time[index], time[index+1], color=color, alpha=0.3)
        ax3.axvspan(time[index], time[index+1], color=color, alpha=0.3)

    # Plot position for each particle
    for i in range(num_particles):
        valid_indices = np.arange(start_rows[i], len(time))  # Time indices where the particle is tracked
        ax1.plot(time[valid_indices], position[valid_indices, i], label=f'Particle {i+1}')

    # Annotate the y-axis for position
    ax1.set_ylabel('Position')
    ax1.legend()

    # Plot velocity for each particle (time array must be shortened by 1)
    for i in range(num_particles):
        valid_indices = np.arange(start_rows[i], len(time) - 1)  # Time indices for velocity are reduced by 1
        ax2.plot(time[valid_indices], velocity[valid_indices, i], label=f'Particle {i+1}')

    # Annotate the y-axis for velocity
    ax2.set_ylabel('Velocity')
    ax2.legend()
    
    # Plot acceleration for each particle (time array must be shortened by 2)
    for i in range(num_particles):
        valid_indices = np.arange(start_rows[i], len(time) - 2)  # Time indices for acceleration are reduced by 2
        ax3.plot(time[valid_indices], acceleration[valid_indices, i], label=f'Particle {i+1}')
    
    # Annotate the y-axis for acceleration
    ax3.set_ylabel('Acceleration')
    ax3.set_xlabel('Time (s)')

    # Set titles and grid
    ax1.set_title('Position, Velocity, and Acceleration over Time')
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

def characterize_motion(velocity, acceleration, velocity_threshold=0.1, acceleration_threshold=0.1):
    """
    Characterize the motion of a particle based on velocity and acceleration.
    
    Parameters:
        velocity (numpy.ndarray): 1D array of velocity values.
        acceleration (numpy.ndarray): 1D array of acceleration values.
        velocity_threshold (float): Threshold below which the velocity is considered "near zero".
        acceleration_threshold (float): Threshold for determining if acceleration is significant.
        
    Returns:
        events (list): List of tuples (motion_type, index), where motion_type describes the motion
                       ('constant_velocity', 'accelerating', 'decelerating', 'turnaround', 'stopped').
    """
    events = []
    for i in range(1, len(velocity) - 1):
        print(i)
        # Detect Turnaround (velocity crosses zero)
        if (velocity[i-1] > 0 and velocity[i] < 0) or (velocity[i-1] < 0 and velocity[i] > 0):
            events.append(('turnaround', i))
        # Detect Constant Velocity (acceleration is near zero)
        elif abs(acceleration[i]) < acceleration_threshold and abs(velocity[i]) >= velocity_threshold:
            events.append(('constant_velocity', i))
        # Detect Accelerating (positive acceleration)
        elif acceleration[i] > acceleration_threshold and abs(velocity[i]) >= velocity_threshold:
            events.append(('accelerating', i))
        # Detect Decelerating (negative acceleration)
        elif acceleration[i] < -acceleration_threshold and abs(velocity[i]) >= velocity_threshold:
            events.append(('decelerating', i))
        # Detect Stopped (velocity near zero)
        elif abs(velocity[i]) < velocity_threshold:
            events.append(('stopped', i))
    return events


if __name__ == "__main__":
    motion_correct = False
    make_kymo = True
    try_particle_tracking = True
    which_dir = '/home/schollab/Documents/autophagy_tal/file_31'
    if motion_correct:
        fr = 30           
        decay_time = 1      
        sparse = False     

        max_shifts = (4, 4)
        strides = (10, 10)
        overlaps = (10, 10)
        max_deviation_rigid = 5

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

        initial_registration(which_dir, 70, mc_dict)
    
    if make_kymo:
        kymo_width = 4
        os.chdir(which_dir)
        assert 'ch2_nonrigid_registered.tif' in os.listdir(), f"ch2 registered data not found for {which_dir}"
        assert 'RoiSet.zip' in os.listdir(), f"roiset not found in {which_dir}"
        stack = tifffile.imread('ch2_nonrigid_registered.tif')
        roi_list = roifile.roiread('RoiSet.zip')
        for i in range(len(roi_list)):
            print(f"making kymo {i} of {len(roi_list)}")
            roi_coords = roi_list[i].coordinates()
            kymograph = create_kymograph(stack, roi_coords, kymo_width)
            # Guassian smooth of 1-2 pixels to smear out things smaller than target
            kymo_mean_row = np.mean(kymograph, axis=0)
            kymo_demeaned_row = np.subtract(kymograph, kymo_mean_row)
            kymo_demean_smooth = gaussian_filter(kymo_demeaned_row, sigma=2, order=0)
            kymo_demean_smooth_thresh = np.copy(kymo_demean_smooth)
            kymo_demean_smooth_thresh[kymo_demean_smooth_thresh < 0] = 0
            kymo_demeaned_first_deriv = np.diff(kymo_demean_smooth_thresh, axis=0)
            if try_particle_tracking:
                initial_conditions, start_rows = select_initial_conditions(kymo_demean_smooth)
                fit_results = track_particles_with_bounds(kymo_demean_smooth, initial_conditions, start_rows, max_mu_shift=3)
                visualize_fit_results(kymo_demean_smooth, fit_results, start_rows, num_particles=len(start_rows))
                position, velocity, acceleration = filter_and_compute_derivatives(fit_results, frame_time_step=1, filter_width=10, start_rows=start_rows)
                

                num_particles = len(start_rows)
                num_frames = kymo_demean_smooth.shape[0]
                data = np.zeros((num_frames, 4 * num_particles)) # posn, velocity, acceleration, characterization
                for p in range(num_particles): 
                    events = characterize_motion(velocity[:,p], acceleration[:,p])
                    code.interact(local=dict(globals(), **locals())) 
                    events_vec = np.zeros((num_frames,))
                    events_vec[0] = np.nan
                    events_vec[-1] = np.nan
                    for i in range(1, num_frames-2):
                        if events[i][0] == 'stopped':
                            events_vec[i] = 0
                        if events[i][0] == 'constant_velocity':
                            events_vec[i] = 1
                    data[:,(4*p)+0] = position[:,p]
                    data[:,(4*p)+1] = velocity[:,p]
                    data[:,(4*p)+2] = acceleration[:,p]
                    #code.interact(local=dict(globals(), **locals())) 
                    data[:,(4*p)+3] = events_vec
                                        # remove sections before tracking
                    data[:start_rows[p],(4*p)+0] = np.nan # pos
                    data[:start_rows[p],(4*p)+1] = np.nan # vel
                    data[:start_rows[p],(4*p)+2] = np.nan # acc
                    data[:start_rows[p],(4*p)+3] = np.nan # events
                code.interact(local=dict(globals(), **locals())) 
                np.savetxt("pva_data.csv", data, delimiter=',')
                    

                # Can only get events from one particle at a time
                #events = characterize_motion(velocity, acceleration)
                # this plot only works for really one particle at a time for observing
                #plot_position_velocity_acceleration(np.arange(0,position.shape[0]), position, velocity, acceleration, start_rows, events)
                
               



"""
Ignore crossing condition merges for now.

Seed the points in a gui before feeding to the fitting function.

Plot these distribution
- velocity (px/frmame)
- acceleration (px/frame)
- number of turns (from the velocity distribution)


Interpolation of the points as well.
"""


