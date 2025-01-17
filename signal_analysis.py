import numpy as np
from scipy.signal import filtfilt, butter
from scipy.stats import linregress

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
        if (((velocity[i-1] > 0 and velocity[i] < 0) or (velocity[i-1] < 0 and velocity[i] > 0) and abs(velocity[i]) <= velocity_threshold)):
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

def better_characterize_motion(velocity, acceleration, velocity_thresh, acc_thresh):
    """
    Labels timepoints for a particle's behavior by frame.

    Parameters:

    Returns:
    """
    n_points = len(velocity)
    event_labels = []

    for i in range(n_points):
        if ((abs(velocity[i]) < velocity_thresh) and (abs(acceleration[i]) > acc_thresh)):
            event_labels.append(('turnaround', i))
        elif (abs(velocity[i]) >= velocity_thresh):
            event_labels.append(('moving', i))
        else:
            event_labels.append(('stopped', i))
    return event_labels

def trackpoints_to_position(trackpoint_array, num_frames):
    """
    Receives coordinates from ImageJ multi-point ROI for one particle.
    Interpolates data between these points.
    Position points before first row and after last row are np.nan

    Parameters:
        trackpoint_array (np.array): nx2 (x-coordinate, y-coordinate) array of points
                                corresponding to the position of a single particle.
        num_frames (int) : Number of timepoints for this FOV.
    Returns:
        y_vals (np.array) : 
    """
    x_vals = np.arange(0, num_frames, 1)
    y_vals = np.interp(x=x_vals, xp=trackpoint_array[:,1], fp=trackpoint_array[:,0])
    first_row = int(np.ceil(trackpoint_array[0,1]))
    last_row = int(np.floor(trackpoint_array[-1,1]))
    y_vals[0:first_row] = np.nan
    y_vals[last_row:] = np.nan
    return y_vals

def characterize_motion_from_coords(coords, movement_threshold=0.15):
    """
    Takes trackpoints coordinates and gives summary statistics.
    
    Parameters:
        coords (np.ndarray): 2d array of (n_points, 2) for x and y coordinates of trackpoints.
        movement_threshold (float): Threshold (pixels/frame) for a section between trackpoints
                                    to consider the particle as moving.
    Returns:
        summary(list): Aggregate of the following summary statistics:
            - [0] prop_moving (float): Proportion of frames particle velocity is superthreshold 
            - [1] prop_still (float): Proportion of frames particle velocity is subthreshold 
            - [2] avg_velocity (float): Average particle velocity over tracking. (pixels/frame)
            - [3] total_movement (float): Total pixel-distance moved by particle over tracking.
            - [4] direction_changes (float): Number of times a particle achieves superthreshold
                                             velocity direction over the couse of tracking.
    """
    dx = np.diff(coords[:,1])
    dy = np.diff(coords[:,0])
    slopes = dy / dx
    moving_frames = 0
    still_frames = 0

    prev_direction = ''
    direction_changes = 0
    for i in range(len(slopes)):
        if np.abs(slopes[i]) > movement_threshold:
            moving_frames += dx[i]
            if slopes[i] > 0:
                if prev_direction == 'neg':
                    direction_changes += 1
                prev_direction = 'pos'
            if slopes[i] < 0:
                if prev_direction == 'pos':
                    direction_changes += 1
                prev_direction = 'neg'
        else:
            still_frames += dx[i]
    prop_moving = moving_frames / dx.sum()
    prop_still = still_frames / dx.sum()
    avg_velocity = np.mean(np.abs(slopes))
    total_movement = np.abs(dy).sum()
    #changes_dir =  (pos_movement and neg_movement)
    
    summary = [prop_moving, prop_still, avg_velocity, total_movement, direction_changes]
    return summary