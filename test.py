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
    return df

def find_farthest_points(df):
    """
    Finds the two farthest points in the DataFrame based on x_coord and y_coord.

    Parameters:
    df (pd.DataFrame): DataFrame with columns ['frame_id', 'x_coord', 'y_coord', 'z_coord']

    Returns:
    tuple: Indices of the two farthest points and the maximum distance between them
    """
    max_distance = 0
    point1, point2 = None, None

    for (index1, row1), (index2, row2) in combinations(df.iterrows(), 2):
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
optimal_angle_dia = find_optimal_rotation(diastolic_data, point1_dia, point2_dia)
diastolic_data_rotated = rotate_points(diastolic_data, optimal_angle_dia)
df_indexed_dia = indexing_points(diastolic_data_rotated)
df_indexed_dia.to_csv('indexed_points_dia.csv', index=False)

# Plot rotated points for diastolic data
plot_rotated_points(diastolic_data_rotated, point1_dia, point2_dia, 'plots_dia', optimal_angle_dia)

# Find farthest points for systolic data
point1_sys, point2_sys, _ = find_farthest_points(systolic_data)
optimal_angle_sys = find_optimal_rotation(systolic_data, point1_sys, point2_sys)
systolic_data_rotated = rotate_points(systolic_data, optimal_angle_sys)
df_indexed_sys = indexing_points(systolic_data_rotated)
df_indexed_sys.to_csv('indexed_points_sys.csv', index=False)

# Plot rotated points for systolic data
plot_rotated_points(systolic_data_rotated, point1_sys, point2_sys, 'plots_sys', optimal_angle_sys)




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
specific_frame_id = 571  # Change this to the frame_id you want to check

# Apply indexing to the diastolic data
df_indexed_dia = indexing_points(diastolic_data)

# Plot indices for the specific frame_id
plot_indices(df_indexed_dia, 'plots_dia_indices', optimal_angle_dia, specific_frame_id)
