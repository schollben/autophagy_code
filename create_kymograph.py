import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import cv2

def create_kymograph(stack, polyline_roi, width=4):
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
    kymo_movie = np.zeros((width, total_distance, num_frames))

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
                kymo_movie[:,current_position + j ,t] = intensities
        
        current_position += segment_length

    return kymograph, kymo_movie

def create_tracking_kymo(kymograph, coords):
    """
    Subtracts mean row of kymograph, then gaussian smooths prior to fitting.
    Makes it easier to track moving particles

    Parameters:
        kymograph (numpy.ndarray): Kymograph as a 2D array (time, polyline length)
        polyline_roi (numpy.ndarray): 2D array (n_points, 2) of (y, x) coordinates defining the polyline.
    
    Returns:
        clean_kymo (numpy.ndarray): Kymograph as a 2D array (time, polyline length),
                    but mean-row subtracted and 2d smoothed, then row derivative.
    """
    kymo_mean_row = np.mean(kymograph, axis=0)
    kymo_mean_subd = np.subtract(kymograph, kymo_mean_row)
    
    kymo_subd_smoothed = gaussian_filter(kymo_mean_subd, sigma=2, order=0)
    
    kymo_threshed = np.copy(kymo_subd_smoothed)
    kymo_threshed[kymo_threshed < 0] = 0
    
    #kymo_derivitive = np.diff(kymo_threshed, axis=0)
    return kymo_threshed

def create_kymo_movie(array, output_filename, framerate=30):
    """
    Converts a (width, length, time) numpy array of float values to a video file.

    Parameters:
        array (numpy.ndarray): 3D array of shape (width, length, time) with float values.
        output_filename (str): Path to save output file ("kymo_roi_3.mp4)
        framerate (int): Framerate of the output video
    Returns:
        None, writes a video file as side-effect.
    """
    array_min = array.min()
    array_max = array.max()
    normalized_array = 255 * (array - array_min) / (array_max - array_min)
    array_uint8 = normalized_array.astype(np.uint8)

    width = array.shape[0]
    height = array.shape[1]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, framerate, (height, width), isColor=False)

    for i in range(array_uint8.shape[2]):
        frame = array_uint8[:,:,i]
        out.write(frame)
    out.release()
    print(f"Kymograph video saved as {output_filename}")