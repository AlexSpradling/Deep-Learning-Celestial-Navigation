import re
import datetime as dt
import numpy as np



def extract_position_time(filepath):
    """
    The labels corresponding to the image are encoded in the title. This function
    parses the labels.

    Args:
    ---
        filepath(str): ex: cloud_cover0/L38.0241LON-138.9065T2020-03-13-04-00-00.png
    
    The string is parsed to extract the latitude, longitude and time of the image.
    """
    # filename = filepath.split("/")[-1]
    filename = filepath

    # extract position
    lat = re.search('L(.*?)LON', filename).group(1)
    long = re.search('LON(.*?)T', filename).group(1)

    # extract time
    time = re.search('T(.*?)\.png', filename).group(1)

    # convert time to np.datetime64
    time = dt.datetime.strptime(time, '%Y-%m-%d-%H-%M-%S')
    time = np.datetime64(time)
    
    # return position and time
    return (float(lat), float(long)), time

def normalize_datetime(time, min_time, max_time):
    """
    For single time used in customgenerator, not array
    """
    # normalize the datetime64 object
    normalized_time = (time - min_time) / (max_time - min_time)
    return normalized_time

def get_lat_long_bounds(y):
    """
    Get the bounds of the latitude and longitude

    Args:
        y (np.array): Array of positions to be used as reference for normalization

    Returns:
        tuple: (lat_min, lat_range, long_min, long_range)
    """
    lat_min = y[:,0].min()
    lat_max = y[:,0].max()
    long_min = y[:,1].min()
    long_max = y[:,1].max()
    lat_range = lat_max - lat_min
    long_range = long_max - long_min

    return lat_min, lat_range, long_min, long_range
  
def normalize_y(pos_array, master_pos):
    """
    Normalize the position array to be between 0 and 1

    Args:
        pos_array (np.array): Array of positions to be normalized
        master_pos (np.array): Array of positions to be used as reference for normalization

    Returns:
        np.array: Normalized positions
    """
    lat_min, lat_range, long_min, long_range = get_lat_long_bounds(master_pos)
    
    y_norm = np.zeros(pos_array.shape)
    
    y_norm[:, 0] = (pos_array[:, 0] - lat_min) / lat_range
    y_norm[:, 1] = (pos_array[:, 1] - long_min) / long_range

    return y_norm

    
