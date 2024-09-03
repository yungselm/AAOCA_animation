import os

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from itertools import combinations

# read indexed_points_dia.csv and indexed_points_sys.csv with headers
df_sys = pd.read_csv('indexed_points_dia.csv', header=0)
df_dia = pd.read_csv('indexed_points_sys.csv', header=0)

# keep only frame_id 0, 1, 2 for both
# df_sys = df_sys[df_sys['frame_id'].isin([0, 1, 2])]
# df_dia = df_dia[df_dia['frame_id'].isin([0, 1, 2])]

df_sys['z_coord'] = np.abs(df_sys['z_coord'] - df_sys['z_coord'].max())
df_dia['z_coord'] = np.abs(df_dia['z_coord'] - df_dia['z_coord'].max())

def calculate_normal_vector(p0, p1, p2):
    v1 = p1 - p0
    v2 = p2 - p0
    normal_vector = np.cross(v1, v2)
    normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)
    return normal_vector_normalized

def calculate_distance(centroid1, centroid2):
    return np.linalg.norm(centroid1 - centroid2)

def define_planes(df_dia, df_sys):
    # Initialize separate result lists for dia and sys
    results_dia = []
    results_sys = []

    # Determine the length of the shorter dataframe
    min_length = min(len(df_dia['frame_id'].unique()), len(df_sys['frame_id'].unique()))

    for i in range(min_length - 1):
        # Get the data for the current frame and the next frame for both dia and sys
        frame_current_dia = df_dia[df_dia['frame_id'] == i]
        frame_next_dia = df_dia[df_dia['frame_id'] == i + 1]
        
        frame_current_sys = df_sys[df_sys['frame_id'] == i]
        frame_next_sys = df_sys[df_sys['frame_id'] == i + 1]

        # Ensure the index is sorted
        frame_current_dia = frame_current_dia.sort_values('index')
        frame_next_dia = frame_next_dia.sort_values('index')
        
        frame_current_sys = frame_current_sys.sort_values('index')
        frame_next_sys = frame_next_sys.sort_values('index')

        # Compute the centroid between the current and next frame for dia and sys
        interframe_centroid_dia = np.mean(
            [frame_current_dia[['centroid_x', 'centroid_y', 'centroid_z']].values, 
             frame_next_dia[['centroid_x', 'centroid_y', 'centroid_z']].values],
            axis=0
        )
        interframe_centroid_sys = np.mean(
            [frame_current_sys[['centroid_x', 'centroid_y', 'centroid_z']].values, 
             frame_next_sys[['centroid_x', 'centroid_y', 'centroid_z']].values],
            axis=0
        )

        # plot the centroid and x_coord, y_coord, z_coord of each frame
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(frame_current_dia['x_coord'], frame_current_dia['y_coord'], frame_current_dia['z_coord'], c='r', marker='o')
        ax.scatter(frame_next_dia['x_coord'], frame_next_dia['y_coord'], frame_next_dia['z_coord'], c='b', marker='o')
        ax.scatter(interframe_centroid_dia[0][0], interframe_centroid_dia[0][1], interframe_centroid_dia[0][2], c='pink', marker='o')
        ax.scatter(frame_current_dia['centroid_x'], frame_current_dia['centroid_y'], frame_current_dia['centroid_z'], c='g', marker='o')
        plt.show()

        # Loop through each index in the current frame except the last one
        for j in range(min(len(frame_current_dia), len(frame_current_sys)) - 1):
            # Process dia
            p0_dia = frame_current_dia.iloc[j][['x_coord', 'y_coord', 'z_coord']].values
            p1_dia = frame_current_dia.iloc[j + 1][['x_coord', 'y_coord', 'z_coord']].values
            p2_dia = frame_next_dia.iloc[j][['x_coord', 'y_coord', 'z_coord']].values
            p3_dia = frame_next_dia.iloc[j + 1][['x_coord', 'y_coord', 'z_coord']].values

            centroid_dia = np.mean([p0_dia, p1_dia, p2_dia, p3_dia], axis=0)
            normal_vector_dia = calculate_normal_vector(p0_dia, p1_dia, p2_dia)

            distance_centroids_dia = calculate_distance(interframe_centroid_dia, centroid_dia)

            results_dia.append({
                'frame_id': i,
                'index': j,
                'points': [p0_dia, p1_dia, p2_dia, p3_dia],
                'centroid': centroid_dia,
                'normal_vector': normal_vector_dia,
                'translation_distances': 0,  # Dia does not require translation distance
                'color': 'grey'
            })

            # Process sys
            p0_sys = frame_current_sys.iloc[j][['x_coord', 'y_coord', 'z_coord']].values
            p1_sys = frame_current_sys.iloc[j + 1][['x_coord', 'y_coord', 'z_coord']].values
            p2_sys = frame_next_sys.iloc[j][['x_coord', 'y_coord', 'z_coord']].values
            p3_sys = frame_next_sys.iloc[j + 1][['x_coord', 'y_coord', 'z_coord']].values

            centroid_sys = np.mean([p0_sys, p1_sys, p2_sys, p3_sys], axis=0)
            normal_vector_sys = calculate_normal_vector(p0_sys, p1_sys, p2_sys)
            distance_centroids_sys = calculate_distance(interframe_centroid_sys, centroid_sys)

            # Calculate translation distance
            translation_distance = 0
            if results_sys:  # Only calculate if there's a previous entry
                translation_distance = calculate_distance(results_sys[-1]['centroid'], centroid_sys)
                if distance_centroids_sys < distance_centroids_dia:
                    translation_distance *= -1

            results_sys.append({
                'frame_id': i,
                'index': j,
                'points': [p0_sys, p1_sys, p2_sys, p3_sys],
                'centroid': centroid_sys,
                'normal_vector': normal_vector_sys,
                'translation_distances': translation_distance,
                'color': None
            })

    return results_dia, results_sys

# Example usage
results_dia, results_sys = define_planes(df_dia, df_sys)
# create and print a pandas dataframe from results_dia and results_sys
results_dia_pd = pd.DataFrame(results_dia)
results_sys_pd = pd.DataFrame(results_sys)
# # duplicate every row in results_dia_pd and results_sys_pd
# results_dia_pd = results_dia_pd.loc[results_dia_pd.index.repeat(2)].reset_index(drop=True)
# results_sys_pd = results_sys_pd.loc[results_sys_pd.index.repeat(2)].reset_index(drop=True)

# def duplicate_every_n_rows(df, n=20):
#     """
#     Duplicates every 'n' consecutive rows immediately after they appear in the DataFrame.
    
#     Parameters:
#         df (pd.DataFrame): Original DataFrame.
#         n (int): Number of consecutive rows to duplicate together.
        
#     Returns:
#         pd.DataFrame: DataFrame with every 'n' rows duplicated consecutively.
#     """
#     # Calculate the number of full groups of 'n' rows
#     num_full_groups = len(df) // n
#     remainder = len(df) % n
    
#     # Initialize list to collect chunks
#     chunks = []
    
#     # Process full groups
#     for i in range(num_full_groups):
#         start_idx = i * n
#         end_idx = start_idx + n
#         group = df.iloc[start_idx:end_idx]
#         # Append the group twice
#         chunks.extend([group, group.copy()])
    
#     # Process remaining rows if any
#     if remainder > 0:
#         remaining_group = df.iloc[num_full_groups * n:]
#         chunks.extend([remaining_group, remaining_group.copy()])
    
#     # Concatenate all chunks into a single DataFrame
#     duplicated_df = pd.concat(chunks, ignore_index=True)
#     return duplicated_df

# results_dia_pd = duplicate_every_n_rows(results_dia_pd)
# results_sys_pd = duplicate_every_n_rows(results_sys_pd)

# write to scv files
results_dia_pd.to_csv('results_dia.csv', index=False)
results_sys_pd.to_csv('results_sys.csv', index=False)

# merge only the trnslation_distances column of results_sys_pd to results_dia_pd with df_dia and df_sys
print(results_dia_pd.head(50))
print(results_sys_pd.head(50))
df_dia['translation_distances'] = results_dia_pd['translation_distances'].values
df_sys['translation_distances'] = results_sys_pd['translation_distances'].values

print(df_dia.head(50))


# # Print the first few results for checking
# print("Dia Results:", results_dia[:2])
# print("Sys Results:", results_sys[:2])

# # print maximum value of translation distance of results_sys
# max_translation_distance = max([result['translation_distances'] for result in results_sys])
# min_translation_distance = min([result['translation_distances'] for result in results_sys])
# print("Max Translation Distance:", max_translation_distance)
# print("Min Translational Distance:", min_translation_distance)


def plot_3d_planes(df):
    """
    Plots 3D planes between corresponding points from consecutive frames.

    Parameters:
    df (pd.DataFrame): DataFrame with coordinates and frame IDs.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    unique_frames = sorted(df['frame_id'].unique())
    
    for i in range(len(unique_frames) - 1):
        frame1_data = df[df['frame_id'] == unique_frames[i]]
        frame2_data = df[df['frame_id'] == unique_frames[i + 1]]
        
        # Sort by index to ensure corresponding points are connected
        frame1_data = frame1_data.sort_values(by='index')
        frame2_data = frame2_data.sort_values(by='index')
        
        for j in range(len(frame1_data)):
            # Create a quadrilateral (two triangles) to form a plane between corresponding points
            v0 = [frame1_data.iloc[j]['x_coord_normalized'], frame1_data.iloc[j]['y_coord_normalized'], frame1_data.iloc[j]['z_coord']]
            v1 = [frame2_data.iloc[j]['x_coord_normalized'], frame2_data.iloc[j]['y_coord_normalized'], frame2_data.iloc[j]['z_coord']]
            v2 = [frame2_data.iloc[(j + 1) % len(frame2_data)]['x_coord_normalized'], frame2_data.iloc[(j + 1) % len(frame2_data)]['y_coord_normalized'], frame2_data.iloc[(j + 1) % len(frame2_data)]['z_coord']]
            v3 = [frame1_data.iloc[(j + 1) % len(frame1_data)]['x_coord_normalized'], frame1_data.iloc[(j + 1) % len(frame1_data)]['y_coord_normalized'], frame1_data.iloc[(j + 1) % len(frame1_data)]['z_coord']]
            
            verts = [v0, v1, v2, v3]
            
            # Add the quadrilateral as a face
            poly = Poly3DCollection([verts], color='skyblue', alpha=0.5)
            ax.add_collection3d(poly)
    
    # Set labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    
    # Ensure the axes have the same scale
    x_limits = [df['x_coord_normalized'].min(), df['x_coord_normalized'].max()]
    y_limits = [df['y_coord_normalized'].min(), df['y_coord_normalized'].max()]
    z_limits = [df['z_coord'].min(), df['z_coord'].max()]
    
    # Find the range for each axis
    ranges = np.array([x_limits, y_limits, z_limits])
    axis_range = [ranges.min(), ranges.max()]
    
    # Apply the same limits to all axes
    ax.set_xlim(axis_range)
    ax.set_ylim(axis_range)
    ax.set_zlim(axis_range)
    
    # Adjust the view angle for better visualization
    ax.view_init(elev=30, azim=120)
    
    plt.show()

# Example usage with the provided DataFrame
# Make sure your DataFrame is loaded as `df`
plot_3d_planes(df_sys)

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# def calculate_plane_center(v0, v1, v2, v3):
#     """
#     Calculates the center of a plane defined by four points.
    
#     Parameters:
#     v0, v1, v2, v3 (list): Coordinates of the four points defining the plane.
    
#     Returns:
#     np.array: The center (centroid) of the plane.
#     """
#     return np.mean([v0, v1, v2, v3], axis=0)

# def calculate_distance(point1, point2):
#     """
#     Calculates the Euclidean distance between two points in 3D space.
    
#     Parameters:
#     point1, point2 (np.array): Coordinates of the two points.
    
#     Returns:
#     float: The Euclidean distance between the points.
#     """
#     return np.linalg.norm(point1 - point2)

# def process_frames(df_sys, df_dia):
#     """
#     Processes systolic and diastolic data to calculate plane translations between consecutive frames.
    
#     Parameters:
#     df_sys, df_dia (pd.DataFrame): Systolic and Diastolic datasets.
    
#     Returns:
#     list: List of distances between systolic and diastolic centroids for each frame.
#     """
#     translation_distances = []
#     unique_frames = sorted(df_sys['frame_id'].unique())

#     for i in range(len(unique_frames) - 1):
#         frame_sys_1 = df_sys[df_sys['frame_id'] == unique_frames[i]]
#         frame_sys_2 = df_sys[df_sys['frame_id'] == unique_frames[i + 1]]

#         frame_dia_1 = df_dia[df_dia['frame_id'] == unique_frames[i]]
#         frame_dia_2 = df_dia[df_dia['frame_id'] == unique_frames[i + 1]]

#         # Sort points by index to ensure correct pairing
#         frame_sys_1 = frame_sys_1.sort_values(by='index')
#         frame_sys_2 = frame_sys_2.sort_values(by='index')

#         frame_dia_1 = frame_dia_1.sort_values(by='index')
#         frame_dia_2 = frame_dia_2.sort_values(by='index')

#         # Loop over corresponding points and calculate plane centers
#         for j in range(len(frame_sys_1) - 1):
#             # Define plane for systolic
#             v0_sys = frame_sys_1.iloc[j][['x_coord_normalized', 'y_coord_normalized', 'z_coord']].values
#             v1_sys = frame_sys_2.iloc[j][['x_coord_normalized', 'y_coord_normalized', 'z_coord']].values
#             v2_sys = frame_sys_2.iloc[(j + 1) % len(frame_sys_2)][['x_coord_normalized', 'y_coord_normalized', 'z_coord']].values
#             v3_sys = frame_sys_1.iloc[(j + 1) % len(frame_sys_1)][['x_coord_normalized', 'y_coord_normalized', 'z_coord']].values

#             center_sys = calculate_plane_center(v0_sys, v1_sys, v2_sys, v3_sys)

#             # Define plane for diastolic
#             v0_dia = frame_dia_1.iloc[j][['x_coord_normalized', 'y_coord_normalized', 'z_coord']].values
#             v1_dia = frame_dia_2.iloc[j][['x_coord_normalized', 'y_coord_normalized', 'z_coord']].values
#             v2_dia = frame_dia_2.iloc[(j + 1) % len(frame_dia_2)][['x_coord_normalized', 'y_coord_normalized', 'z_coord']].values
#             v3_dia = frame_dia_1.iloc[(j + 1) % len(frame_dia_1)][['x_coord_normalized', 'y_coord_normalized', 'z_coord']].values

#             center_dia = calculate_plane_center(v0_dia, v1_dia, v2_dia, v3_dia)

#             # Calculate the distance between systolic and diastolic centers
#             translation_distance = calculate_distance(center_sys, center_dia)
#             translation_distances.append(translation_distance)
    
#     return translation_distances

# def plot_translation_distances(translation_distances, output_dir='plots'):
#     """
#     Plots the translation distances over frame indices.
    
#     Parameters:
#     translation_distances (list): List of distances between systolic and diastolic centroids for each frame.
#     output_dir (str): Directory to save the plot.
#     """
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(len(translation_distances)), translation_distances, marker='o', linestyle='-', color='b')
#     plt.title('Translation Distances Between Systolic and Diastolic Planes')
#     plt.xlabel('Frame Index')
#     plt.ylabel('Translation Distance')
#     plt.grid(True)

#     # Save plot
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     plt.savefig(os.path.join(output_dir, 'translation_distances.png'))
#     plt.show()

# # Example usage
# translation_distances = process_frames(df_sys, df_dia)
# plot_translation_distances(translation_distances)