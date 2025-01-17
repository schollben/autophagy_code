import numpy as np
import matplotlib.pyplot as plt


def visualize_fit_results(kymograph, fit_results, start_rows):
    """
    Visualize the fit results by plotting the kymograph and marking the mu positions
    of the Gaussian fits for each particle with small circles.
    
    Parameters:
        kymograph (numpy.ndarray): 2D array of the kymograph image (rows=time, cols=space).
        fit_results (numpy.ndarray): 2D array of the fit results (rows=time, cols=3*num_particles).
        start_rows (list of int): List of row indices where tracking of each particle begins.
    """
    num_particles = len(start_rows)
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
        ax1.axvspan(time[index], time[index], color=color, alpha=0.3)
        ax2.axvspan(time[index], time[index], color=color, alpha=0.3)
        ax3.axvspan(time[index], time[index], color=color, alpha=0.3)

    # Plot position for each particle
    for i in range(num_particles):
        valid_indices = np.arange(start_rows[i], len(time))  # Time indices where the particle is tracked
        ax1.plot(time[valid_indices], position[valid_indices, i], label=f'Particle {i+1}')

    # Annotate the y-axis for position
    ax1.set_ylabel('Position')
    ax1.legend()

    # Plot velocity for each particle (time array must be shortened by 1)
    for i in range(num_particles):
        valid_indices = np.arange(start_rows[i], len(time))  # Time indices for velocity are reduced by 1
        ax2.plot(time[valid_indices], velocity[valid_indices, i], label=f'Particle {i+1}')

    # Annotate the y-axis for velocity
    ax2.set_ylabel('Velocity')
    ax2.legend()
    
    # Plot acceleration for each particle (time array must be shortened by 2)
    for i in range(num_particles):
        valid_indices = np.arange(start_rows[i], len(time))  # Time indices for acceleration are reduced by 2
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