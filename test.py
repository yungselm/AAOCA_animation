import numpy as np
import pandas as pd
import os
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

# def rotate_points(df, point1, point2):
#     """
#     Rotates all points so that the line between the two farthest points is vertical.

#     Parameters:
#     df (pd.DataFrame): DataFrame with columns ['frame_id', 'x_coord', 'y_coord', 'z_coord']
#     point1, point2 (int): Indices of the two farthest points

#     Returns:
#     pd.DataFrame: DataFrame with rotated x and y coordinates
#     """
#     # Get the coordinates of the farthest points
#     x1, y1 = df.loc[point1, ['x_coord', 'y_coord']]
#     x2, y2 = df.loc[point2, ['x_coord', 'y_coord']]
    
#     # Calculate the angle to rotate
#     theta = np.arctan2(y2 - y1, x2 - x1)
    
#     # Adjust the angle by 90 degrees to make the line vertical
#     # theta += np.pi / 4
    
#     # Calculate the rotation matrix
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)
    
#     # Rotate all points
#     df['x_coord_rotated'] = df['x_coord'] * cos_theta - df['y_coord'] * sin_theta
#     df['y_coord_rotated'] = df['x_coord'] * sin_theta + df['y_coord'] * cos_theta
    
#     return df


# def plot_rotated_points(df, point1, point2, output_dir):
#     """
#     Plots the rotated points and the line between the farthest points.

#     Parameters:
#     df (pd.DataFrame): DataFrame with rotated coordinates.
#     point1, point2 (int): Indices of the two farthest points.
#     output_dir (str): Directory to save the plots.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     for frame_id in df['frame_id'].unique():
#         frame_data = df[df['frame_id'] == frame_id]
        
#         plt.scatter(frame_data['x_coord_rotated'], frame_data['y_coord_rotated'], label='Rotated Points')
        
#         # Plot line between the farthest points
#         plt.plot([df.loc[point1, 'x_coord_rotated'], df.loc[point2, 'x_coord_rotated']],
#                  [df.loc[point1, 'y_coord_rotated'], df.loc[point2, 'y_coord_rotated']],
#                  color='red', label='Farthest Points')
        
        
#         plt.title(f'Frame ID: {frame_id}')
#         plt.legend()

#         plt.gca().set_aspect('equal', adjustable='box')
#         plt.savefig(os.path.join(output_dir, f'frame_{frame_id}.png'))
#         plt.close()

# # Main workflow
# diastolic_data = read_data('test_csv_files/diastolic_contours.csv')
# systolic_data = read_data('test_csv_files/systolic_contours.csv')

# # For every frame_id keep only 20 evenly spaced points
# diastolic_data = diastolic_data.groupby('frame_id').apply(lambda x: x.iloc[::int(np.ceil(len(x)/20))])
# systolic_data = systolic_data.groupby('frame_id').apply(lambda x: x.iloc[::int(np.ceil(len(x)/20))])

# # Find farthest points for diastolic data
# point1_dia, point2_dia, max_dist_dia = find_farthest_points(diastolic_data)
# diastolic_data_rotated = rotate_points(diastolic_data, point1_dia, point2_dia)
# plot_rotated_points(diastolic_data_rotated, point1_dia, point2_dia, 'plots_dia')

# # Find farthest points for systolic data
# point1_sys, point2_sys, max_dist_sys = find_farthest_points(systolic_data)
# systolic_data_rotated = rotate_points(systolic_data, point1_sys, point2_sys)
# plot_rotated_points(systolic_data_rotated, point1_sys, point2_sys, 'plots_sys')

# print(diastolic_data_rotated.head())

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

# Main workflow
diastolic_data = pd.read_csv('test_csv_files/diastolic_contours.csv', delimiter='\t', header=None)
diastolic_data.columns = ['frame_id', 'x_coord', 'y_coord', 'z_coord']
systolic_data = pd.read_csv('test_csv_files/systolic_contours.csv', delimiter='\t', header=None)
systolic_data.columns = ['frame_id', 'x_coord', 'y_coord', 'z_coord']

# For every frame_id keep only 20 evenly spaced points
diastolic_data = diastolic_data.groupby('frame_id').apply(lambda x: x.iloc[::int(np.ceil(len(x)/20))])
systolic_data = systolic_data.groupby('frame_id').apply(lambda x: x.iloc[::int(np.ceil(len(x)/20))])

# Find farthest points for diastolic data
point1_dia, point2_dia, _ = find_farthest_points(diastolic_data)
optimal_angle_dia = find_optimal_rotation(diastolic_data, point1_dia, point2_dia)
plot_rotated_points(diastolic_data, point1_dia, point2_dia, 'plots_dia', optimal_angle_dia)

# Find farthest points for systolic data
point1_sys, point2_sys, _ = find_farthest_points(systolic_data)
optimal_angle_sys = find_optimal_rotation(systolic_data, point1_sys, point2_sys)
plot_rotated_points(systolic_data, point1_sys, point2_sys, 'plots_sys', optimal_angle_sys)

print(diastolic_data.head())
