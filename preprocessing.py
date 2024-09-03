# import os
# import hydra
# from omegaconf import DictConfig

# import numpy as np
# import pandas as pd
# pd.options.mode.chained_assignment = None  # default='warn'
# import warnings
# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# warnings.simplefilter(action='ignore', category=DeprecationWarning)
# import matplotlib.pyplot as plt
# from itertools import combinations

# @hydra.main(version_base=None, config_path='.', config_name='config')
# class Preprocessing:
#     """ Takes coordinates from IVUS frames, and reduces the number of points, rotates the contour
#     and further translates the whole contour to the centroid.
    
#     Parameters: 
#     Input path, output path and number of points per contour can be defined in the config
#     file. 
    
#     Returns: Dataframe with adjusted coordinates."""

#     def __init__(self, config: DictConfig) -> None:
#         self.config = config
#         self.diastolic_data = read_data(self.config.preprocessing.diastolic_df)
#         self.systolic_data = read_data(self.config.preprocessing.systolic_df)
#         self.diastolic_data = self.diastolic_data.reset_index()
#         self.systolic_data = self.systolic_data.reset_index()

#     def __call__(self, config: DictConfig) -> None:
#         self.diastolic_data = pd.read_csv(self.config.preprocessing.redcap_database)
#         self.systolic_data = pd.read_csv(self.config.preprocessing.redcap_database)
#         dia_df = self.diastolic_data.groupby('frame_id').apply(lambda x: x.iloc[::int(np.ceil(len(x)/config.preprocessing.n_points))]).reset_index(drop=True)
#         sys_df = self.systolic_data.groupby('frame_id').apply(lambda x: x.iloc[::int(np.ceil(len(x)/config.preprocessing.n_points))]).reset_index(drop=True)

        
#         return self.diastolic_data, self.systolic_data

# # For every frame_id keep only 20 evenly spaced points
# diastolic_data = 
# systolic_data = systolic_data.groupby('frame_id').apply(lambda x: x.iloc[::int(np.ceil(len(x)/20))]).reset_index(drop=True)

# # Find farthest points for diastolic data
# point1_dia, point2_dia, _ = find_farthest_points(diastolic_data)
# optimal_angle_dia = find_optimal_rotation(diastolic_data, point1_dia, point2_dia)
# diastolic_data_rotated = rotate_points(diastolic_data, optimal_angle_dia)
# df_indexed_dia = indexing_points(diastolic_data_rotated)
# # normalize to centroid
# diastolic_data_with_centroids = calculate_centroid(diastolic_data_rotated)
# diastolic_data_normalized = normalize_to_centroid(diastolic_data_with_centroids)
# diastolic_data_normalized.to_csv('indexed_points_dia.csv', index=False)
# # Plot rotated points for diastolic data
# plot_rotated_points(diastolic_data_rotated, point1_dia, point2_dia, 'plots_dia', optimal_angle_dia)

# # Find farthest points for diastolic data
# point1_sys, point2_sys, _ = find_farthest_points(systolic_data)
# optimal_angle_sys = find_optimal_rotation(systolic_data, point1_sys, point2_sys)
# systolic_data_rotated = rotate_points(systolic_data, optimal_angle_sys)
# df_indexed_sys = indexing_points(systolic_data_rotated)
# # normalize to centroid
# systolic_data_with_centroids = calculate_centroid(systolic_data_rotated)
# systolic_data_normalized = normalize_to_centroid(systolic_data_with_centroids)
# systolic_data_normalized.to_csv('indexed_points_sys.csv', index=False)
# # Plot rotated points for diastolic data
# plot_rotated_points(systolic_data_rotated, point1_sys, point2_sys, 'plots_dia', optimal_angle_sys)

import os

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import matplotlib.pyplot as plt
from itertools import combinations

def read_data(file_path, delimiter='\t'):
    """
    Reads the data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.
    delimiter (str): Delimiter used in the CSV file.

    Returns:
    pd.DataFrame: DataFrame with the loaded data.
    """
    df = pd.read_csv(file_path, delimiter=delimiter, header=None)
    df.columns = ['frame_id', 'x_coord', 'y_coord', 'z_coord']

    # Renumber frame_ids
    unique_frame_ids = sorted(df['frame_id'].unique(), reverse=True)
    frame_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_frame_ids)}
    df['frame_id'] = df['frame_id'].map(frame_id_mapping)

    return df

def find_farthest_points(frame_data):
    """
    Finds the two farthest points within a single frame based on x_coord and y_coord.

    Parameters:
    frame_data (pd.DataFrame): DataFrame with columns ['x_coord', 'y_coord']

    Returns:
    tuple: Indices of the two farthest points and the maximum distance between them
    """
    max_distance = 0
    point1, point2 = None, None

    for (index1, row1), (index2, row2) in combinations(frame_data.iterrows(), 2):
        distance = np.sqrt((row1['x_coord'] - row2['x_coord'])**2 + (row1['y_coord'] - row2['y_coord'])**2)
        if distance > max_distance:
            max_distance = distance
            point1, point2 = index1, index2

    return point1, point2, max_distance

def rotate_points(df, angle):
    """
    Rotates all points by the given angle.

    Parameters:
    df (pd.DataFrame): DataFrame with columns ['x_coord', 'y_coord']
    angle (float): Angle in radians to rotate the points

    Returns:
    pd.DataFrame: DataFrame with rotated x and y coordinates
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    df['x_coord_rotated'] = df['x_coord'] * cos_angle - df['y_coord'] * sin_angle
    df['y_coord_rotated'] = df['x_coord'] * sin_angle + df['y_coord'] * cos_angle
    
    return df

def process_frame(frame_data):
    point1, point2, _ = find_farthest_points(frame_data)
    optimal_angle = find_optimal_rotation(frame_data, point1, point2)
    rotated_frame_data = rotate_points(frame_data, optimal_angle)
    return rotated_frame_data, point1, point2, optimal_angle

def process_all_frames(df):
    results = []
    for frame_id, frame_data in df.groupby('frame_id'):
        rotated_frame_data, point1, point2, optimal_angle = process_frame(frame_data)
        results.append((rotated_frame_data, point1, point2, optimal_angle))
    return results

def x_coord_distance(df, point1, point2):
    """
    Calculates the x-coordinate distance between two points in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame with rotated coordinates
    point1, point2 (int): Indices of the two points

    Returns:
    float: The x-coordinate distance between the two points
    """
    return abs(df.loc[point1, 'x_coord_rotated'] - df.loc[point2, 'x_coord_rotated'])

def find_optimal_rotation(df, point1, point2, angle_step=np.pi/180):
    """
    Finds the optimal rotation angle that minimizes the x-coordinate distance between two points.

    Parameters:
    df (pd.DataFrame): DataFrame with columns ['x_coord', 'y_coord']
    point1, point2 (int): Indices of the two points
    angle_step (float): Step size for the angle in radians

    Returns:
    float: The optimal rotation angle in radians
    """
    min_distance = float('inf')
    best_angle = 0
    
    for angle in np.arange(0, 2*np.pi, angle_step):
        rotated_df = rotate_points(df.copy(), angle)
        distance = x_coord_distance(rotated_df, point1, point2)
        
        if distance < min_distance:
            min_distance = distance
            best_angle = angle
            
    return best_angle

def indexing_points(df):
    """
    Sets indices of points in a clockwise manner with the point with the highest y-coordinate being 0,
    for each unique frame_id separately.

    Parameters:
    df (pd.DataFrame): DataFrame with columns ['x_coord_rotated', 'y_coord_rotated']

    Returns:
    pd.DataFrame: DataFrame with new 'index' column
    """
    def process_frame(frame_data):
        # Find the point with the highest y-coordinate
        highest_point = frame_data.loc[frame_data['y_coord_rotated'].idxmax()]

        # Calculate angles for sorting points
        def calculate_angle(row):
            return np.arctan2(row['y_coord_rotated'] - highest_point['y_coord_rotated'],
                              row['x_coord_rotated'] - highest_point['x_coord_rotated'])
        
        # Calculate angles and sort points
        frame_data['angle'] = frame_data.apply(calculate_angle, axis=1)
        sorted_df = frame_data.sort_values(by='angle', ascending=False).reset_index(drop=True)
        
        # Assign indices starting from 0 and increasing clockwise
        sorted_df['index'] = range(len(sorted_df))
        return sorted_df
    
    # Apply processing to each frame_id
    df_indexed = df.groupby('frame_id').apply(process_frame).reset_index(drop=True)
    
    return df_indexed

def calculate_centroid(df):
    """
    Calculates the centroid of points for each frame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the coordinates and frame_id.

    Returns:
    pd.DataFrame: DataFrame with centroid_x and centroid_y columns added.
    """
    centroids = df.groupby('frame_id').apply(lambda x: pd.Series({
        'centroid_x': x['x_coord_rotated'].mean(),
        'centroid_y': x['y_coord_rotated'].mean(),
        'centroid_z': x['z_coord'].mean()
    })).reset_index()
    
    return pd.merge(df, centroids, on='frame_id')

def normalize_to_centroid(df):
    """
    Normalizes the coordinates to the centroid for each frame.

    Parameters:
    df (pd.DataFrame): DataFrame with original and centroid coordinates.

    Returns:
    pd.DataFrame: DataFrame with normalized coordinates.
    """
    df['x_coord_normalized'] = df['x_coord_rotated'] - df['centroid_x']
    df['y_coord_normalized'] = df['y_coord_rotated'] - df['centroid_y']
    df['z_coord_normalized'] = df['z_coord']
    
    return df

def plot_rotated_points(df, point1, point2, output_dir, angle):
    """
    Plots the rotated points and the line between the farthest points with equal scaling for the axes.

    Parameters:
    df (pd.DataFrame): DataFrame with rotated coordinates.
    point1, point2 (int): Indices of the two farthest points.
    output_dir (str): Directory to save the plots.
    angle (float): Angle used for rotation in radians.
    """
    df_rotated = rotate_points(df, angle)
    
    os.makedirs(output_dir, exist_ok=True)

    for frame_id in df_rotated['frame_id'].unique():
        frame_data = df_rotated[df_rotated['frame_id'] == frame_id]
        
        plt.scatter(frame_data['x_coord_rotated'], frame_data['y_coord_rotated'], label='Rotated Points')
        
        # Plot line between the farthest points
        plt.plot([df_rotated.loc[point1, 'x_coord_rotated'], df_rotated.loc[point2, 'x_coord_rotated']],
                 [df_rotated.loc[point1, 'y_coord_rotated'], df_rotated.loc[point2, 'y_coord_rotated']],
                 color='red', label='Farthest Points')
        
        plt.title(f'Frame ID: {frame_id}')
        plt.legend()
        
        # Set equal scaling for x- and y-axis
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.savefig(os.path.join(output_dir, f'frame_{frame_id}.png'))
        plt.close()


# Read data
diastolic_data = read_data('test_csv_files/diastolic_contours.csv')
systolic_data = read_data('test_csv_files/systolic_contours.csv')

# Ensure frame_id is a column for grouping
diastolic_data = diastolic_data.reset_index()
systolic_data = systolic_data.reset_index()

# For every frame_id keep only 20 evenly spaced points
diastolic_data = diastolic_data.groupby('frame_id').apply(lambda x: x.iloc[::int(np.ceil(len(x)/20))]).reset_index(drop=True)
systolic_data = systolic_data.groupby('frame_id').apply(lambda x: x.iloc[::int(np.ceil(len(x)/20))]).reset_index(drop=True)

# Find farthest points for diastolic data
point1_dia, point2_dia, _ = find_farthest_points(diastolic_data)
optimal_angle_dia = find_optimal_rotation(diastolic_data, point1_dia, point2_dia) # prorbably mistake by not grouping by frame id
diastolic_data_rotated = rotate_points(diastolic_data, optimal_angle_dia)
df_indexed_dia = indexing_points(diastolic_data_rotated)
# normalize to centroid
diastolic_data_with_centroids = calculate_centroid(df_indexed_dia)
diastolic_data_normalized = normalize_to_centroid(diastolic_data_with_centroids)
diastolic_data_normalized.to_csv('indexed_points_dia.csv', index=False)
# Plot rotated points for diastolic data
plot_rotated_points(diastolic_data_rotated, point1_dia, point2_dia, 'plots_dia', optimal_angle_dia)

# Find farthest points for diastolic data
point1_sys, point2_sys, _ = find_farthest_points(systolic_data)
optimal_angle_sys = find_optimal_rotation(systolic_data, point1_sys, point2_sys)
systolic_data_rotated = rotate_points(systolic_data, optimal_angle_sys)
df_indexed_sys = indexing_points(systolic_data_rotated)
# normalize to centroid
systolic_data_with_centroids = calculate_centroid(df_indexed_sys)
systolic_data_normalized = normalize_to_centroid(systolic_data_with_centroids)
systolic_data_normalized.to_csv('indexed_points_sys.csv', index=False)
# Plot rotated points for diastolic data
plot_rotated_points(systolic_data_rotated, point1_sys, point2_sys, 'plots_dia', optimal_angle_sys)

# Process diastolic data
diastolic_results = process_all_frames(diastolic_data)
diastolic_processed_frames = [result[0] for result in diastolic_results]
diastolic_data_rotated = pd.concat(diastolic_processed_frames)
diastolic_data_rotated = indexing_points(diastolic_data_rotated)
diastolic_data_with_centroids = calculate_centroid(diastolic_data_rotated)
diastolic_data_normalized = normalize_to_centroid(diastolic_data_with_centroids)
diastolic_data_normalized.to_csv('indexed_points_dia.csv', index=False)

# Plot each frame's results
for frame_result in diastolic_results:
    rotated_frame_data, point1, point2, optimal_angle = frame_result
    plot_rotated_points(rotated_frame_data, point1, point2, 'plots_dia', optimal_angle)

# Repeat for systolic data
systolic_results = process_all_frames(systolic_data)
systolic_processed_frames = [result[0] for result in systolic_results]
systolic_data_rotated = pd.concat(systolic_processed_frames)
systolic_data_rotated = indexing_points(systolic_data_rotated)
systolic_data_with_centroids = calculate_centroid(systolic_data_rotated)
systolic_data_normalized = normalize_to_centroid(systolic_data_with_centroids)
systolic_data_normalized.to_csv('indexed_points_sys.csv', index=False)

# Plot each frame's results
for frame_result in systolic_results:
    rotated_frame_data, point1, point2, optimal_angle = frame_result
    plot_rotated_points(rotated_frame_data, point1, point2, 'plots_sys', optimal_angle)


def plot_indices(df, output_dir, angle, frame_id):
    """
    Plots the indices of points in a specific frame with equal scaling for the axes.

    Parameters:
    df (pd.DataFrame): DataFrame with rotated coordinates and indices.
    output_dir (str): Directory to save the plots.
    angle (float): Angle used for rotation in radians.
    frame_id (int): Specific frame_id to plot.
    """
    # Apply rotation
    df_rotated = rotate_points(df, angle)
    
    # Filter data for the specific frame_id
    frame_data = df_rotated[df_rotated['frame_id'] == frame_id]
    
    # Create a new directory for plots if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot indices
    plt.figure(figsize=(8, 8))
    
    # Plot indices as text
    for _, row in frame_data.iterrows():
        plt.text(row['x_coord_rotated'], row['y_coord_rotated'], str(int(row['index'])),
                 fontsize=12, ha='right', color='blue')

    # Set equal scaling for x- and y-axis
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    
    # Optionally set limits to ensure everything fits
    padding = 1.0
    x_min, x_max = frame_data['x_coord_rotated'].min() - padding, frame_data['x_coord_rotated'].max() + padding
    y_min, y_max = frame_data['y_coord_rotated'].min() - padding, frame_data['y_coord_rotated'].max() + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.title(f'Frame ID: {frame_id} - Indexed Points')
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'frame_{frame_id}_indexed.png'))
    plt.close()

# Assuming the DataFrames and indexing are already done
# Select a specific frame_id to plot
specific_frame_id = 0  # Change this to the frame_id you want to check

# Apply indexing to the diastolic data
df_indexed_dia = indexing_points(diastolic_data)

# Plot indices for the specific frame_id
plot_indices(df_indexed_dia, 'plots_dia_indices', optimal_angle_dia, specific_frame_id)
