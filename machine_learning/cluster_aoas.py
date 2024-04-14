import h5py
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os
from typing import Callable
from cprint import *
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
# from dataset_consumer import DatasetConsumer
from torch.utils.data import Dataset, DataLoader

DATASET = './machine_learning/data/dataset_0_5m_spacing.h5'
# # # d.print_info()

# # # # Start with curved paths
# paths = d.generate_straight_paths(2, path_length_n=5)

# # paths = d.generate_straight_paths(1)
# aoas = d.ray_aoas
# mags_fromloss = d.ray_path_losses

# print(d.weighted_aoa_average(paths).shape)
# d.paths_to_dataset_mag_rays_weighted_aoas(paths)
# d.weighted_aoa_average(paths)
# print(paths.shape[1])
# print(paths.shape)
# print(paths)
# mags = d.paths_to_dataset_mag_only(paths)
# mags_paths = d.paths_to_dataset_mag_plus_rays(paths)
# print(mags)
# num_rays = d.get_num_rays(paths)
# print(d.paths_to_dataset_mag_only(paths).shape)
# print(num_rays)
# print("######################")

# aoa_ray_mag = d.paths_to_dataset_mag_rays_aoas(paths) # returns a tuple (azimuths, elevations)


# # Create left, center, and right paths
# left_paths, center_paths, right_paths = d.create_left_center_right_paths(paths, terminal_length=10)

# # Create datasets from the paths
# left_dataset = d.paths_to_dataset_interleaved_w_relative_positions(left_paths)
# center_dataset = d.paths_to_dataset_interleaved_w_relative_positions(center_paths)
# right_dataset = d.paths_to_dataset_interleaved_w_relative_positions(right_paths)

# # Plot the first 50 paths' relative positions
# for i in range(50):
#     plt.plot(left_dataset[i, :, -1], left_dataset[i, :, -2])
#     plt.plot(right_dataset[i, :, -1], right_dataset[i, :, -2])
#     plt.plot(center_dataset[i, :, -1], center_dataset[i, :, -2])
# plt.show()



class AoAClusteredDataset(Dataset):

    def __init__(self, path_indices, dataset_path):
        self.ray_aoas = None
        self.ray_path_losses = None

        with h5py.File(dataset_path, 'r') as file:
            self.ray_aoas = file['ray_aoas'][:]
            self.ray_path_losses = file['ray_path_losses'][:]

        self.path_indices = path_indices
        print("Setting up dataset of aoa clusters averaged with weights of magnitudes...")
        self.aoas = self.ray_aoas
        self.mags_fromloss = self.ray_path_losses
        self.weighted_aoa_pts = self.weighted_aoa_set(self.path_indices)
        print("Complete.")

    def __len__(self):
        return len(self.path_indices)
    
    def __getitem__(self, idx):
        # pts = self.path_indices[idx]
        # weighted_aoa_pt = self.weighted_aoa_set(path_indices)
        return self.weighted_aoa_pts[idx]
        # return weighted_aoa_pt[idx]

    def weighted_aoa_set(self, pts):
        """
        Generate a torch dataset from the given path indices to contain
        the mags, number of rays, and aoa average 
        Shape: (num_paths, path_length_n, 128)
        """    
        def convert_negative_to_positive(angles):
            # Convert angles to degrees
            angles_degrees = np.rad2deg(angles)
            # Convert negative angles to positive within the range [0, 360)
            converted_angles_degrees = np.where(angles_degrees < 0, 360 + angles_degrees, angles_degrees)
            # Convert back to radians
            converted_angles_radians = np.radians(converted_angles_degrees)
            return converted_angles_radians
    
        n_avgs_per_pt = 10 # highest number of clusters to introduce padding
        weighted_aoa_paths = []
        
        temp_num_pts = 0
        # for path in path_indices:
        #     temp_num_paths += 1
        weighted_aoa_pts = np.empty((0, n_avgs_per_pt))  # Initialize an empty array of points for each path
        for pt in pts:
            temp_num_pts+=1
            print(temp_num_pts)
            mags_sample = np.trim_zeros(self.mags_fromloss[:, pt])
            aoas_sample = np.trim_zeros(np.deg2rad(self.aoas[:, 0, pt]))

            converted_aoas_angles = convert_negative_to_positive(aoas_sample)  # Wrap angles

            # Reshape the wrapped angles
            data_aoas = converted_aoas_angles.reshape(-1, 1)

            ########
            # Isolation Forest - Isolating the outlier points and getting the inliers
            ########
            if(data_aoas.size > 0):
                # Define Isolation Forest model
                isolation_forest = IsolationForest()

                # Fit the model to the data
                isolation_forest.fit(data_aoas)

                # Predict outliers (-1) and inliers (1)
                cluster_labels = isolation_forest.predict(data_aoas)

                # Get inlier indices 
                inlier_indices = np.where(cluster_labels == 1)

                # Get the data values (mags and aoas) for the linear indices
                data_aoas_inliers = data_aoas[inlier_indices]
                aoas_inliers = converted_aoas_angles[inlier_indices]
                mags_inliers = mags_sample[inlier_indices]

                ########
                # Apply the DBSCAN clustering to the dataset with outliers removed
                ########
                if(data_aoas_inliers.size > 0):
                    # Perform clustering with DBSCAN on the "wrapped" angles, from 0 to 360 degrees
                    dbscan_aoa = DBSCAN(eps=0.1, min_samples=3)
                    aoa_cluster_labels = dbscan_aoa.fit_predict(data_aoas_inliers)  # if any of the aoa clusters are negative then they aren't part of a group

                    # Get each of the clusters, note that negative labels are outliers and should be removed
                    unique_labels = np.unique(aoa_cluster_labels)
                    unique_labels = unique_labels[np.where(unique_labels >= 0)]
                    cluster_averages = []

                    # Graphing the points:
                    # Plot clustered data
                    # aoaos_inliers = converted_aoas_angles[inlier_indices]
                    # mags_inliers = mags_sample[inlier_indices]
                    # fig = plt.subplots(subplot_kw=dict(projection="polar"))
                    # # print(aoa_cluster_labels)
                    # # print(np.where(aoa_cluster_labels < 0))
                    # plt.scatter(aoaos_inliers, mags_inliers, c=aoa_cluster_labels)
                    # plt.title('Clustered Data, only AoAs (DBSCAN)')
                    # plt.xlabel('AoAs')
                    # plt.ylabel('Magnitudes')
                    # plt.colorbar(label='Cluster Label')
                    # plt.savefig('cluster_plots/on_paths/test-plot+point{}+path{}.png'.format(pt, temp_num_paths))

                    # find the weighted average of each of the clusters and append to cluster_averages
                    for label in unique_labels:
                        cluster = np.where(aoa_cluster_labels == label)
                        weighted_average = np.average(np.rad2deg(aoas_inliers[cluster]), weights=mags_inliers[cluster])
                        cluster_averages.append(weighted_average)
                        
                    # Convert cluster_averages to a numpy array
                    cluster_averages = np.array(cluster_averages)
                    # Adding padding to ensure each cluster of averages has 4 elements,
                    padding_needed = n_avgs_per_pt - len(cluster_averages)
                    # print("path %d", temp_num_paths)
                    # print("avg per pt", n_avgs_per_pt)
                    # print("clusters", len(cluster_averages))
                    # print(padding_needed)
                    padded_cluster_averages = np.pad(cluster_averages, (0, padding_needed), mode='constant') # Adding zero to the extra values
            
            # Append padded cluster averages to the weighted_aoa_pt array
            weighted_aoa_pts = np.vstack((weighted_aoa_pts, padded_cluster_averages))
        
        # Append the weighted_aoa_pt for this path to the main list
        # weighted_aoa_paths.append(weighted_aoa_pt)
        return weighted_aoa_pts

# Generate path indices from 0 to 40400
# path_indices = range(52)
# custom_dataset = AoAClusteredDataset(path_indices, DATASET)

# # paths = d.generate_straight_paths(2,2)
# paths = [[50, 45],[51, 30]]
# print(paths)
# # Create an instance of CustomDataset
# # print(custom_dataset.__getitem__(22))
# print(custom_dataset.__getitem__(paths))
# Create a DataLoader instance
# train_dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# # Iterate over the DataLoader
# for batch in train_dataloader:
#     # Process each batch as needed
#     passz