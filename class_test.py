import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from itertools import combinations

# Suppress warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

class Preprocessing:
    """
    Preprocessing IVUS contour data by reducing points, rotating contours, 
    and normalizing to the centroid.

    Parameters:
    config (dict): Configuration dictionary containing input paths, output paths, 
                   and parameters like number of points per contour.
    """

    def __init__(self, config):
        self.config = config
        self.diastolic_data = self.read_data(self.config['preprocessing']['diastolic_df'])
        self.systolic_data = self.read_data(self.config['preprocessing']['systolic_df'])
        self.diastolic_data = self.diastolic_data.reset_index()
        self.systolic_data = self.systolic_data.reset_index()

    def __call__(self):
        """ Calls the process_data method when the instance is called. """
        self.process_data()

    def read_data(self, file_path, delimiter='\t'):
        """ Reads the data from a CSV file and renumbers frame_ids. """
        df = pd.read_csv(file_path, delimiter=delimiter, header=None)
        df.columns = ['frame_id', 'x_coord', 'y_coord', 'z_coord']
        unique_frame_ids = sorted(df['frame_id'].unique(), reverse=True)
        frame_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_frame_ids)}
        df['frame_id'] = df['frame_id'].map(frame_id_mapping)
        return df

    def find_farthest_points(self, frame_data):
        """ Finds the two farthest points within a single frame. """
        max_distance = 0
        point1, point2 = None, None
        for (index1, row1), (index2, row2) in combinations(frame_data.iterrows(), 2):
            distance = np.sqrt((row1['x_coord'] - row2['x_coord'])**2 + (row1['y_coord'] - row2['y_coord'])**2)
            if distance > max_distance:
                max_distance = distance
                point1, point2 = index1, index2
        return point1, point2, max_distance

    def rotate_points(self, df, angle):
        """ Rotates all points by the given angle. """
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        df['x_coord_rotated'] = df['x_coord'] * cos_angle - df['y_coord'] * sin_angle
        df['y_coord_rotated'] = df['x_coord'] * sin_angle + df['y_coord'] * cos_angle
        return df

    def find_optimal_rotation(self, df, point1, point2, angle_step=np.pi/180):
        """ Finds the optimal rotation angle that minimizes the x-coordinate distance between two points. """
        min_distance = float('inf')
        best_angle = 0
        for angle in np.arange(0, 2*np.pi, angle_step):
            rotated_df = self.rotate_points(df.copy(), angle)
            distance = self.x_coord_distance(rotated_df, point1, point2)
            if distance < min_distance:
                min_distance = distance
                best_angle = angle
        return best_angle

    def x_coord_distance(self, df, point1, point2):
        """ Calculates the x-coordinate distance between two points in the DataFrame. """
        return abs(df.loc[point1, 'x_coord_rotated'] - df.loc[point2, 'x_coord_rotated'])

    def indexing_points(self, df):
        """ Sets indices of points in a clockwise manner with the point with the highest y-coordinate being 0. """
        df_indexed = df.groupby('frame_id').apply(self._process_frame_indexing).reset_index(drop=True)
        return df_indexed

    def _process_frame_indexing(self, frame_data):
        highest_point = frame_data.loc[frame_data['y_coord_rotated'].idxmax()]
        frame_data['angle'] = frame_data.apply(lambda row: np.arctan2(row['y_coord_rotated'] - highest_point['y_coord_rotated'],
                                                                      row['x_coord_rotated'] - highest_point['x_coord_rotated']), axis=1)
        sorted_df = frame_data.sort_values(by='angle', ascending=False).reset_index(drop=True)
        sorted_df['index'] = range(len(sorted_df))
        return sorted_df

    def calculate_centroid(self, df):
        """ Calculates the centroid of points for each frame. """
        centroids = df.groupby('frame_id').apply(lambda x: pd.Series({
            'centroid_x': x['x_coord_rotated'].mean(),
            'centroid_y': x['y_coord_rotated'].mean(),
            'centroid_z': x['z_coord'].mean()
        })).reset_index()
        return pd.merge(df, centroids, on='frame_id')

    def normalize_to_centroid(self, df):
        """ Normalizes the coordinates to the centroid for each frame. """
        df['x_coord_normalized'] = df['x_coord_rotated'] - df['centroid_x']
        df['y_coord_normalized'] = df['y_coord_rotated'] - df['centroid_y']
        df['z_coord_normalized'] = df['z_coord']
        return df

    def plot_rotated_points(self, df, point1, point2, output_dir, angle):
        """ Plots the rotated points and the line between the farthest points. """
        df_rotated = self.rotate_points(df, angle)
        os.makedirs(output_dir, exist_ok=True)
        for frame_id in df_rotated['frame_id'].unique():
            frame_data = df_rotated[df_rotated['frame_id'] == frame_id]
            plt.scatter(frame_data['x_coord_rotated'], frame_data['y_coord_rotated'], label='Rotated Points')
            plt.plot([df_rotated.loc[point1, 'x_coord_rotated'], df_rotated.loc[point2, 'x_coord_rotated']],
                     [df_rotated.loc[point1, 'y_coord_rotated'], df_rotated.loc[point2, 'y_coord_rotated']],
                     color='red', label='Farthest Points')
            plt.title(f'Frame ID: {frame_id}')
            plt.legend()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(os.path.join(output_dir, f'frame_{frame_id}.png'))
            plt.close()

    def plot_indices(self, df, output_dir, angle, frame_id):
        """ Plots the indices of points in a specific frame with equal scaling for the axes. """
        df_rotated = self.rotate_points(df, angle)
        frame_data = df_rotated[df_rotated['frame_id'] == frame_id]
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(8, 8))
        for _, row in frame_data.iterrows():
            plt.text(row['x_coord_rotated'], row['y_coord_rotated'], str(int(row['index'])),
                     fontsize=12, ha='right', color='blue')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        padding = 1.0
        ax.set_xlim(frame_data['x_coord_rotated'].min() - padding, frame_data['x_coord_rotated'].max() + padding)
        ax.set_ylim(frame_data['y_coord_rotated'].min() - padding, frame_data['y_coord_rotated'].max() + padding)
        plt.title(f'Frame ID: {frame_id} - Indexed Points')
        plt.savefig(os.path.join(output_dir, f'frame_{frame_id}_indexed.png'))
        plt.close()

    def process_all_frames(self, df):
        """ Processes all frames to rotate points and normalize them. """
        results = []
        for frame_id, frame_data in df.groupby('frame_id'):
            point1, point2, _ = self.find_farthest_points(frame_data)
            optimal_angle = self.find_optimal_rotation(frame_data, point1, point2)
            rotated_frame_data = self.rotate_points(frame_data, optimal_angle)
            results.append((rotated_frame_data, point1, point2, optimal_angle))
        return results

    def process_data(self):
        """ Main method to process diastolic and systolic data and save results. """
        # For every frame_id keep only 20 evenly spaced points
        self.diastolic_data = self.diastolic_data.groupby('frame_id').apply(lambda x: x.iloc[::int(np.ceil(len(x)/20))]).reset_index(drop=True)
        self.systolic_data = self.systolic_data.groupby('frame_id').apply(lambda x: x.iloc[::int(np.ceil(len(x)/20))]).reset_index(drop=True)

        # Process diastolic data
        diastolic_results = self.process_all_frames(self.diastolic_data)
        diastolic_processed_frames = [result[0] for result in diastolic_results]
        diastolic_data_rotated = pd.concat(diastolic_processed_frames)
        diastolic_data_indexed = self.indexing_points(diastolic_data_rotated)
        diastolic_data_with_centroids =         self.calculate_centroid(diastolic_data_indexed)
        diastolic_data_normalized = self.normalize_to_centroid(diastolic_data_with_centroids)
        diastolic_data_normalized.to_csv('indexed_points_dia.csv', index=False)

        # Process systolic data
        systolic_results = self.process_all_frames(self.systolic_data)
        systolic_processed_frames = [result[0] for result in systolic_results]
        systolic_data_rotated = pd.concat(systolic_processed_frames)
        systolic_data_indexed = self.indexing_points(systolic_data_rotated)
        systolic_data_with_centroids = self.calculate_centroid(systolic_data_indexed)
        systolic_data_normalized = self.normalize_to_centroid(systolic_data_with_centroids)
        systolic_data_normalized.to_csv('indexed_points_sys.csv', index=False)

        # Visualization (Optional)
        if self.config['preprocessing']['plot']:
            for result in diastolic_results:
                rotated_frame_data, point1, point2, optimal_angle = result
                self.plot_rotated_points(rotated_frame_data, point1, point2, self.config['preprocessing']['diastolic_output_dir'], optimal_angle)
                self.plot_indices(rotated_frame_data, self.config['preprocessing']['diastolic_output_dir'], optimal_angle, rotated_frame_data['frame_id'].iloc[0])

            for result in systolic_results:
                rotated_frame_data, point1, point2, optimal_angle = result
                self.plot_rotated_points(rotated_frame_data, point1, point2, self.config['preprocessing']['systolic_output_dir'], optimal_angle)
                self.plot_indices(rotated_frame_data, self.config['preprocessing']['systolic_output_dir'], optimal_angle, rotated_frame_data['frame_id'].iloc[0])

# # Configuration dictionary (example)
# config = {
#     'preprocessing': {
#         'diastolic_df': "C:/WorkingData/Documents/2_Coding/Python/AAOCA_animation/test_csv_files/diastolic_contours.csv",
#         'systolic_df': "C:/WorkingData/Documents/2_Coding/Python/AAOCA_animation/test_csv_files/systolic_contours.csv",
#         'diastolic_output_path': "C:/WorkingData/Documents/2_Coding/Python/AAOCA_animation",
#         'systolic_output_path': "C:/WorkingData/Documents/2_Coding/Python/AAOCA_animation",
#         'diastolic_output_dir': 'plots_dia',
#         'systolic_output_dir': 'plots_sys',
#         'plot': True
#     }
# }

# # Instantiate and call the Preprocessing class
# preprocessing = Preprocessing(config)
# preprocessing()
