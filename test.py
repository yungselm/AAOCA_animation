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

print(df_dia.head(50))
print(df_sys.head(50))
print(df_dia.columns)

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
    
    # Adjust the view angle for better visualization
    ax.view_init(elev=30, azim=120)
    
    plt.show()

# Example usage with the provided DataFrame
# Make sure your DataFrame is loaded as `df`
plot_3d_planes(df_dia)

