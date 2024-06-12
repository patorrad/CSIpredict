"""
This contains the AoAClusteredDataset class
"""
import h5py
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os
import time
from typing import Callable
from cprint import *
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from torch.utils.data import Dataset, DataLoader

DATASET = './machine_learning/data/dataset_0_5m_spacing.h5'

class AoAClusteredDataset(Dataset):
    """
    This class sets up the Angle of Arrival (AoA) dataset by clustering all angles of rays recieved 
    at each position in the MATLAB generated dataset and determines each clusters weighted average. 
    """

    def __init__(self, path_indices, dataset_path):
        """
        Initialization of class -- handles the creation of the dataset with weighted clusters
        """
        self.ray_aoas = None
        self.ray_path_losses = None

        with h5py.File(dataset_path, 'r') as file:
            self.ray_aoas = file['ray_aoas'][:]
            self.ray_path_losses = file['ray_path_losses'][:]

        self.path_indices = path_indices
        print("Setting up dataset of aoa clusters averaged with weights of magnitudes...")
        start_time=time.time()
        self.aoas = self.ray_aoas
        self.mags_fromloss = self.ray_path_losses
        self.weighted_aoa_pts, self.weighted_mag_pts = self.weighted_aoa_set(self.path_indices)
        end_time=time.time()
        print("time (s):",end_time-start_time)
        print("Complete.")

    def __len__(self):
        """
        Returns length of the paths
        """
        return len(self.path_indices)
    
    def __getitem__(self, idx):
        """
        Returns the specified item, angle of arrival, by index
        """
        return self.weighted_aoa_pts[idx]

    def __getmags__(self, idx):
        """
        Returns the specified mags by index
        """
        return self.weighted_mag_pts
    
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
    
        n_avgs_per_pt = 11 # highest number of clusters to introduce padding
        weighted_aoa_paths = []
        
        temp_num_pts = 0
        # for path in path_indices:
        #     temp_num_paths += 1
        weighted_aoa_pts = np.empty((0, n_avgs_per_pt))  # Initialize an empty array of points for each path, fill with aoas
        weighted_mag_pts = np.empty((0, n_avgs_per_pt)) # initialize an empty array for the points on each path, fill with mags
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
                # Note -- not setting any additional parameters, but this can be revisited
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
                    # Note -- setting parameters for closeness and minimum # of samples to be 
                    # considered a cluster. Can be revisited.
                    dbscan_aoa = DBSCAN(eps=0.1, min_samples=3)

                     # if any of the AOA clusters are negative then they are not part of the group
                    aoa_cluster_labels = dbscan_aoa.fit_predict(data_aoas_inliers) 

                    # Get each of the clusters. Note that negative labels are outliers and should be removed
                    unique_labels = np.unique(aoa_cluster_labels)
                    unique_labels = unique_labels[np.where(unique_labels >= 0)]
                    cluster_averages = []
                    cluster_mag_averages = []

                    ## Optional - Graphing the points to plot clustered data 
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

                    # Find the weighted average of each cluster and append to cluster_averages
                    for label in unique_labels:
                        cluster = np.where(aoa_cluster_labels == label)
                        weighted_average = np.average(np.rad2deg(aoas_inliers[cluster]), weights=mags_inliers[cluster])
                        weighted_magnitude = np.average(mags_inliers[cluster], weights = np.rad2deg(aoas_inliers[cluster]))
                        cluster_averages.append(weighted_average)
                        cluster_mag_averages.append(weighted_magnitude)

                    # Convert cluster_averages to a numpy array
                    cluster_averages = np.array(cluster_averages)
                    cluster_mag_averages = np.array(cluster_mag_averages)

                    # Adding padding to ensure each cluster of averages has 4 elements,
                    padding_needed = n_avgs_per_pt - len(cluster_averages)
                    padded_cluster_averages = np.pad(cluster_averages, (0, padding_needed), mode='constant') # Adding zero to the extra values
                    padded_cluster_mag_averages = np.pad(cluster_mag_averages, (0, padding_needed), mode='constant') # Adding zero to the extra values
            # Append padded cluster averages to the weighted_aoa_pt array
            weighted_aoa_pts = np.vstack((weighted_aoa_pts, padded_cluster_averages))
            weighted_mag_pts = np.vstack((weighted_mag_pts, padded_cluster_mag_averages))

        # Append the weighted_aoa_pt for this path to the main list
        # weighted_aoa_paths.append(weighted_aoa_pt)
        return weighted_aoa_pts, weighted_mag_pts


# Following uses the class above to create the new clustered angle of arrival dataset
# Generate path indices from 0 to 40400
# The final dataset is on drive

path_indices = range(40401)
weighted_clusters_dataset = AoAClusteredDataset(path_indices, DATASET) 
aoa_weighted_clusters = weighted_clusters_dataset.__getaoa__(path_indices)
mags_weighted_clusters = weighted_clusters_dataset.__getmags__(path_indices)
hf = h5py.File('./machine_learning/data/average_aoa_and_mags_cluster_data.h5', 'w')
hf.create_dataset('aoa_weighted_clusters', data=aoa_weighted_clusters)
hf.create_dataset('average_magnitude_clusters', data=mags_weighted_clusters)
hf.close()