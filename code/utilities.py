import re
import datetime as dt
import numpy as np
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse


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

def normalize_times(times_array, master_times):
    """
    Times are datetime64 objects and are min max scaled to normalize the time values
    between 0 and 1

    args
    ---
    times(np.array) - A numpy array of datetime64 objects.

    """
    times = np.array(times_array)
    times = (times - master_times.min()) / (master_times.max() - master_times.min())
    return times

    
def un_normalize(pos, lat_range, lat_min, long_range, long_min):
  pos_norm = pos
  pos_norm[:,0] = pos_norm[:,0] * lat_range + lat_min
  pos_norm[:,1] = pos_norm[:,1] * long_range + long_min
  
  return pos_norm

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    # taken directly from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_y * 2, height=ell_radius_x * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(-45) \
        .scale(scale_y, scale_x) \
        .translate(mean_y, mean_x)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def mercator_distance(pos1, pos2):
    """
    Compute the distance in nm between two positions on the earth's surface using mercator sailing.

    Args:
        pos1 (tuple): latitude and longitude of first position
        pos2 (tuple): latitude and longitude of second position

    Returns:
        float: distance in nm
    """

    lat1, long1 = pos1
    lat2, long2 = pos2
    # compute meridional parts
    mpartsinitial = 7915.7045 * np.log10(
                np.tan(np.pi / 4 + (np.deg2rad(lat1) / 2))) - 23.2689 * np.sin(
                np.deg2rad(lat1))
    mpartssecond = 7915.7045 * np.log10(np.tan(np.pi / 4 + (np.deg2rad(lat2) / 2))) - 23.2689 * np.sin(
                np.deg2rad(lat2))
                
    little1 = (mpartssecond - mpartsinitial) 
    
    # compute dlat
    dlat = np.deg2rad(lat2 - lat1)

    # compute dlong
    dlong = np.deg2rad(long2 - long1)

    # compute course
    course = np.arctan2(dlong,np.deg2rad(little1/60))

    # compute distance
    d = np.rad2deg(dlat/np.cos(course)) * 60

    return d

