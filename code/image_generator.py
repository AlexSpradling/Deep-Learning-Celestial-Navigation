import numpy as np
import os
import datetime as dt
import random
from pytz import utc
import math

import matplotlib as mpl
mpl.rcParams['savefig.pad_inches'] = 0
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import seaborn as sns

from skyfield.api import Star, load
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
from skyfield.data import hipparcos
from skyfield.projections import build_stereographic_projection

# custom 
from utilities import confidence_ellipse

# use dark background
plt.style.use('dark_background')

# skyfield ephemeris data
eph = load('de421.bsp')
sun = eph['sun']
earth = eph['earth']


def generate_clouds(ax, cloud_color='grey', cloud_cover = 0/8):
    """
    Generates a random number of clouds in a given sky.
    ---
    Parameters:
    ax: matplotlib axis
        The axis to plot the clouds on.
    cloud_color: str
        The color of the clouds.
    cloud_cover: float
        The fraction of the sky covered by clouds.
    ---
    Returns:
    ax: matplotlib axis
    """
    # generate random number of clouds
    if cloud_cover == 0/8 or cloud_cover == 1/8:
        n_clouds = 0
    elif cloud_cover == 2/8:
        n_clouds = np.random.randint(1, 20)
    elif cloud_cover  == 3/8:
        n_clouds = np.random.randint(25, 50)
    elif cloud_cover == 4/8:
        n_clouds = np.random.randint(75,100)
    elif cloud_cover == 5/8:
        n_clouds = np.random.randint(100,200)
    elif cloud_cover == 6/8:
        n_clouds = np.random.randint(200, 300)
    elif cloud_cover == 7/8:
        n_clouds = np.random.randint(300, 500)
    elif cloud_cover == 8/8:
        n_clouds = np.random.randint(500, 1000)

    # generate random x and y coordinates
    x = np.random.uniform(-1, 1, n_clouds)
    y = np.random.uniform(-1, 1, n_clouds)

    # generate random radii
    cloud_size = 0.15
    cloud_alpha = .9
    radii = np.random.uniform(0.01, cloud_size, n_clouds)

    # create a list of circles
    circles = [Circle((x[i], y[i]), radii[i]) for i in range(n_clouds)]
    
    # create a collection of circles
    collection = PatchCollection(circles, facecolor=cloud_color, edgecolor=cloud_color, alpha=cloud_alpha)

    # add collection to the plot
    ax.add_collection(collection)
    
    # remove the border
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')

    return ax

def plot_sky(t, observer, field_of_view_degrees=135, limiting_magnitude=3.5, cloud_cover=1/8, 
              img_directory = '../content/drive/MyDrive/parallel_track_skies/', save_images=True):
    """
    Plots a sky image at a given time and location.

    Args:
        t (Skyfield.time.Timescale): List of times to plot.
        observer (Skyfield observer object): Location of observer.
        field_of_view_degrees (int, optional): Defaults to 135.  
        limiting_magnitude (float, optional): _description_. Defaults to 3.5.
        cloud_cover (int, optional): Random cloud cover. Defaults to 1/8. 8/8 is full cloud cover.
        img_directory (str, optional): _description_. Defaults to '../content/drive/MyDrive/parallel_track_skies/'.
        save_images (bool, optional): _description_. Defaults to True.

    Returns:
        .png image: An image of the sky at the given time and location.
    """

    # code adapted from https://rhodesmill.org/skyfield/plotting-stars.html

    # An ephemeris from the JPL provides Sun and Earth positions.
    p = observer.at(t)

    # look South (180 degrees) and up (90 degrees)
    q = p.from_altaz(alt_degrees=90, az_degrees=180)
   
    # Build a stereographic projection centered on the observer.
    projection = build_stereographic_projection(q)

    # The Hipparcos mission provides our star catalog.
    with load.open(hipparcos.URL) as f:
        stars = hipparcos.load_dataframe(f)

    # Compute the star positions in the observer's frame.
    star_positions = (earth+observer).at(t).observe(Star.from_dataframe(stars))
    stars['x'], stars['y'] = projection(star_positions)

    # remove stars below limiting magnitude
    bright_stars = (stars.magnitude <= limiting_magnitude)
    magnitude = stars['magnitude'][bright_stars]

    # make size of star proportional to magnitude
    marker_size = (0.5 + limiting_magnitude - magnitude) ** 2.0
  
    fig, ax = plt.subplots(figsize=[9,9])

    # Draw the stars.
    ax.scatter(stars['x'][bright_stars], stars['y'][bright_stars],
               s=marker_size, color='white', alpha=.7)

    # add moon
    moon = eph['moon']
    moon_position = (earth+observer).at(t).observe(moon)
    moon_x, moon_y = projection(moon_position)
    ax.scatter(moon_x, moon_y, s=100, color='white')

    # field of view
    angle = np.pi - field_of_view_degrees / 360.0 * np.pi
    limit = np.sin(angle) / (1.0 - np.cos(angle))

    # set limits
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_aspect(1.0)

    # add clouds
    ax = generate_clouds(ax, cloud_cover=cloud_cover)
    
    # remove the border
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    #format as string
    title = f'{observer.latitude.degrees} {observer.longitude.degrees} {t.utc_iso()}'
 
    if save_images != False:
        plt.savefig(img_directory + 'L'+str(round(observer.latitude.degrees, 4)) +'LON'+str(round(observer.longitude.degrees, 4)) + 'T' + t.utc_strftime('%Y-%m-%d-%H-%M-%S') + '.png', pad_inches = None, bbox_inches='tight')
        plt.close(fig)

        del fig
        del ax
    else:
        return fig, ax


def great_circle_track(start, end, n):
    """
    Calculates the great circle track between two points on a sphere.

    Args:
        start (tuple): (latitude, longitude) of the starting point in degrees.
        end (tuple): (latitude, longitude) of the ending point in degrees.
        n (int): Number of points along the great circle track.

    Returns:
        list: List of (latitude, longitude) tuples along the great circle track.
    """
    # Convert latitude and longitude to radians
    lat1, lon1 = math.radians(start[0]), math.radians(start[1])
    lat2, lon2 = math.radians(end[0]), math.radians(end[1])

    # Calculate the difference between the longitudes
    dlon = lon2 - lon1

    # Calculate the great circle track using the Haversine formula
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = 3440.069 * c  # 3440.069 is the conversion factor from kilometers to nautical miles

    # Calculate the bearing (angle) in radians
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(y, x)

    # Create a list to hold the points along the track
    points = []

    # Calculate the points along the track
    for i in range(n+1):
        fraction = i/n
        A = math.sin((1-fraction)*c) / math.sin(c)
        B = math.sin(fraction*c) / math.sin(c)
        x = A * math.cos(lat1) * math.cos(lon1) + B * math.cos(lat2) * math.cos(lon2)
        y = A * math.cos(lat1) * math.sin(lon1) + B * math.cos(lat2) * math.sin(lon2)
        z = A * math.sin(lat1) + B * math.sin(lat2)
        lat = math.atan2(z, math.sqrt(x**2 + y**2))
        lon = math.atan2(y, x)
        points.append((math.degrees(lat), math.degrees(lon)))

    return points


def parallel_tracks(track, n):
    """
    Calculates the parallel tracks displaced by n nautical miles from the original track.

    Args:
        track (list): List of (latitude, longitude) tuples along the original track.
        n (int): Number of nautical miles to displace the track.

    Returns:
        tuple: Tuple of lists of (latitude, longitude) tuples along the north and south tracks.
    """

    # Convert n from nautical miles to radians
    n_radians = n / 3440.069

    # Create lists to hold the points for the north and south tracks
    north_track = []
    south_track = []

    # Shift the track points north and south
    for point in track:
        lat, lon = point[0], point[1]

        # Shift the point north
        lat_north = lat + math.degrees(n_radians)
        north_track.append((lat_north, lon))

        # Shift the point south
        lat_south = lat - math.degrees(n_radians)
        south_track.append((lat_south, lon))

    return north_track, south_track


def get_waypoints(start, end, n, list_of_mile_displacements):
    """
    Uses the great circle track and parallel tracks functions to calculate the waypoints for the training data.
    
    Args:
        start (tuple): (latitude, longitude) of the starting point in degrees.
        end (tuple): (latitude, longitude) of the ending point in degrees.
        n (int): Number of points along the great circle track.
        list_of_mile_displacements (list): List of nautical mile displacements for the parallel tracks.

    Returns:
        list: List of (latitude, longitude) tuples along the great circle track and parallel tracks.
    """

    # main trackline
    track = great_circle_track(start, end, n)

    # get displaced tracklines
    displaced_tracks = []
    for displacement in list_of_mile_displacements:
        north_track, south_track = parallel_tracks(track, displacement)
        displaced_tracks.append(north_track)
        displaced_tracks.append(south_track)
    
    # concatenate all tracks
    all_tracks = np.concatenate([track] + displaced_tracks, axis=0)
    
    #remove duplicates
    all_tracks = np.unique(all_tracks, axis=0)

    print('Number of unique waypoints: ', len(all_tracks))
    return all_tracks


def create_times_array(start_time, end_time, n):
    """
    Creates a list of evenly spaced datetime objects between start_time and end_time.

    Args:
        start_time (datetime): Start time.
        end_time (datetime): End time.
        n (int): Number of minutes between each time.

    Returns:
        tuple: Tuple of lists of datetime objects and datetime objects localized to UTC.
    """

    # create a list of evenly spaced datetime objects
    delta = dt.timedelta(minutes=n)
    time_list = []
    while start_time <= end_time:
        time_list.append(start_time)
        start_time += delta


    # localize the times to UTC
    time_list_utc = [utc.localize(t) for t in time_list]

    ts = load.timescale()
    times = ts.from_datetimes(time_list_utc)
    return times, time_list_utc


class DRCalc():
    """ 
    ** This is a modified version of a much more robust class that I wrote for my library Capella. **
    ** Check it out at : https://github.com/AlexSpradling/Capella.git **
    Class to calculate the coordinates of a point given a starting point, a time, a course, and a speed.

    Args:
        init_lat (float): Initial latitude in degrees.
        init_long (float): Initial longitude in degrees.
        timedelta (float): Time in hours.
        course (float): Course in degrees.
        speed (float): Speed in knots.
    
    """
    
    def __init__(self, init_lat, init_long, timedelta, course, speed):
        self.init_lat = float(init_lat)
        self.init_long = float(init_long)
        self.timedelta = float(timedelta) / 3600
        self.course = float(course)
        self.speed = float(speed)
        self.dr_coord_calc_fwd()
  
        return

    def dr_coord_calc_fwd(self):
        self.distance = self.timedelta * self.speed
        if self.course == 90:
            self.lat2 = self.init_lat
            self.dlo = (self.distance / np.cos(np.deg2rad(self.init_lat))) / 60
        elif self.course == 270:
            self.lat2 = self.init_lat
            self.dlo = -1 * (self.distance / np.cos(np.deg2rad(self.init_lat))) / 60
        else:
            if 0 < self.course < 90:
                self.courseangle = self.course
            elif 90 < self.course < 180:
                self.courseangle = 180 - self.course
            elif 180 < self.course < 270:
                self.courseangle = self.course + 180
            else:
                self.courseangle = 360 - self.course
            self.lat2 = (self.distance * np.cos(np.deg2rad(self.course))) / 60 + self.init_lat
            mpartsinitial = 7915.7045 * np.log10(
                np.tan(np.pi / 4 + (np.deg2rad(self.init_lat) / 2))) - 23.2689 * np.sin(
                np.deg2rad(self.init_lat))
            mpartssecond = 7915.7045 * np.log10(np.tan(np.pi / 4 + (np.deg2rad(self.lat2) / 2))) - 23.2689 * np.sin(
                np.deg2rad(self.lat2))
            littlel = mpartssecond - mpartsinitial
            self.dlo = (littlel * np.tan(np.deg2rad(self.course))) / 60
        self.drlatfwds = self.lat2
        self.drlongfwds = self.init_long + self.dlo
        if self.drlongfwds >= 180:
            self.drlongfwds = self.drlongfwds - 360

        return


def show_training_grid(start, end, points_along_track=100, displacements=10, displacement_interval=1):
    """
    Plots a grid of points along a great circle track and parallel tracks. This is just a convenience function for visualizing the training data.

    Args:
        start (tuple): Starting coordinates (latitude, longitude).
        end (tuple): Ending coordinates (latitude, longitude).
        points_along_track (int, optional): Number of points along the great circle track. Defaults to 100.
        displacements (int, optional): Number of parallel tracks. Defaults to 10.
        displacement_interval (int, optional): Interval between parallel tracks. Defaults to 1.
    """

    flat_track = get_waypoints(start, end, points_along_track, [x for x in range(0,displacements,displacement_interval)])
     
    # plot points in flat_track
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(flat_track[:, 1], flat_track[:, 0], s=1, c='red')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Training Grid')

    # use sns darkgrid style
    sns.set_style('darkgrid')

    # show plot
    plt.show()

    return 

def plot_lat_long(y_pred, y_true, df, show_track = False):
  
    mpl.rcParams['savefig.pad_inches'] = 0.2
    """
    Plots the predicted and actual positions.

    Args:
        y_pred (numpy.ndarray): Predicted positions.
        y_true (numpy.ndarray): Actual positions.
        df (pandas.DataFrame): DataFrame containing the training data.
        show_track (bool, optional): Whether or not to show the track. Defaults to False.
    """

    # use seaborn darkgrid
    sns.set_style("darkgrid")

    # create figure and axes
    fig, ax = plt.subplots(figsize=(8,8))

    if show_track !=False:
        # get start and end coordinates
        start = (39, -140)
        end = (37, -138)

        # get waypoints
        waypoints = get_waypoints(start, end, 100, list_of_mile_displacements=[x for x in range(0, 10, 1)])

        # use '+' symbol for waypoints
        ax.plot(waypoints[:,1], waypoints[:,0], markersize=1, label='Waypoints', alpha = 0.2)

    # plot actual position
    ax.scatter(y_true[:, 1],y_true[:, 0], label='actual', color = 'red', marker='x')
 
    # plot predicted positions
    for i in range(len(y_pred)):

        # plot individual predictions
        ax.scatter(y_pred[i][:,0][:,1], y_pred[i][:,0][:,0], color='blue', alpha=0.05, s = 10)

        # plot mean
        ax.scatter(y_pred[i][:,0][:,1].mean(), y_pred[i][:,0][:,0].mean(), color='blue', alpha=1.0, marker='o', s=10)

        # plot confidence ellipse
        confidence_ellipse(y_pred[i][:,0][:,0], y_pred[i][:,0][:,1], ax, n_std=2.0, edgecolor='green', linestyle=':')

        # add time label
        ax.text(y_pred[i][:,0][:,1].mean() + .005, y_pred[i][:,0][:,0].mean(), df['time'][i], fontsize=8)
        
    # labels
    ax.legend(['Actual', 'Monte Carlo Positions','Predicted', 'Two Sigma Ellipse'])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Predicted vs. Actual Position Prediction use Monte Carlo Dropout')

    return 

