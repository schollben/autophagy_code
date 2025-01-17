import os
import shutil

def organize_imaging_files(data_dir, prefix):
    """
    Simply utility code to get the data organized.
    Takes a data directory and a prefix.
    Detects the FOVs within that directory, then creates a directory in 'data_dir' matching the prefix and FOV number.
    Moves tif files that match prefix and FOV number into their respective folders.

    Parameters:
        data_dir (str): Parent directory of the data to seek.
        prefix (str): Prefix of the files to process + prefix for directory names.
    Returns:
        Nothing, does disk operations.
    """
    # Get all files in the data directory that start with the specified prefix
    files = [f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith('.tif')]
    
    for file in files:
        # Extract the FOV number from the filename by splitting on underscores and taking the last part before ".tif"
        fov_number = file.split('_')[-1].split('.')[0]
        
        # Create a directory with the prefix added before "FOV"
        fov_dir = os.path.join(data_dir, f"{prefix}_FOV_{fov_number}")
        os.makedirs(fov_dir, exist_ok=True)
        
        # Move the file to the FOV directory
        source_path = os.path.join(data_dir, file)
        dest_path = os.path.join(fov_dir, file)
        shutil.move(source_path, dest_path)
        print(f"Moved {file} to {fov_dir}")