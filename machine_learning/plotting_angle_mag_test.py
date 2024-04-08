import h5py
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import os
from typing import Callable
from cprint import *
from dataset_consumer import DatasetConsumer

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

DATASET = './machine_learning/data/dataset_0_5m_spacing.h5'
d = DatasetConsumer(DATASET)
# d.print_info()

mags = d.csi_mags
aoas = d.ray_aoas
mags_fromloss = d.ray_path_losses

# Graph 1 - Plotting the first set of rays magnitude & aoa ray pairs for positions 0 to 10 
# EXAMPLE: mags_1 = mags[0,:10] # first set of rays, magnitudes at first 10 positions
mags_1 = mags_fromloss[0, :10]
#print(mags_1)

aoas_1 = np.deg2rad(aoas[0,0,:10]) # first set of rays, azimuth, first 10 positions
#print(np.rad2deg(aoas_1))

# Plot in polar coordinates, changing min and max of radial coordinates based on magnitudes
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
scatter = ax.scatter(aoas_1, mags_1, c=mags_1, cmap='viridis') # plotting the polar coordinates as a scatter plot
ax.set_rmin(np.min(mags_1))
ax.set_rmax(np.max(mags_1) + 1)

# Add colorbar for reference - purple to yellow, purple is lower magnitude
cbar = plt.colorbar(scatter, ax=ax, label='Magnitude Color Reference')
plt.savefig('plot_mags_aoas/first10positions_ray0_fig_1.png')

# Graph 2 - Plotting aoa for first 10 rays at position 0 with path loss as the magnitude
mags_2 = mags_fromloss[:10, 0]
#print(mags_2)

aoas_2 = np.deg2rad(aoas[:10,0,0]) # first set of rays, azimuth, first 10 positions
#print(np.rad2deg(aoas_2))

# Plot in polar coordinates, changing min and max of radial coordinates based on magnitude range
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
scatter = ax.scatter(aoas_2, mags_2, c=mags_2, cmap='viridis') # plotting the polar coordinates as a scatter plot
ax.set_rmin(np.min(mags_2))
ax.set_rmax(np.max(mags_2) + 1)

# Add colorbar for reference - purple to yellow, purple is lower magnitude
cbar = plt.colorbar(scatter, ax=ax, label='Magnitude Color Reference')
plt.savefig('plot_mags_aoas/first10rays_position0_fig_2.png')

# Graph 3 - Generate ONE straight path and plot the points on it - note that each position can have up
#           to 100 rays but will likely be less. Each point represents the magnitude and aoa of a ray.

num_points = 5 # number of points on path, indicates the different positions
paths = d.generate_straight_paths(1, num_points)
#print("PATHS")
#print(paths)

mags_3 = d.paths_to_dataset_path_loss_only(paths)
#print(mags_3.shape) # Shape: (1,num_points,100) -> 1 path, num_points, up to 100 rays per point/position

aoas_3 = d.paths_to_dataset_rays_aoas(paths)[0]
#print(aoas_3)

# Setting up plot of polar coordinates, width of saved figure will change based on number of points
fig, ax = plt.subplots(1, num_points, subplot_kw=dict(projection="polar"),figsize=(28, num_points*3))
# ax.set_rmin(np.min(mags_3[0,1,:]))
# ax.set_rmax(np.max(mags_3[0,1,:]) + 1)
for n in range(num_points):
    random_colormap = np.random.choice([cm.Purples, cm.Greens, cm.Reds, cm.Blues, cm.Oranges])
    # Create a polar graph with all 100 rays (may have fewer) for n'th position
    scatter = ax[n].scatter(aoas_3[0,n,:], mags_3[0,n,:], c=mags_3[0,n,:], cmap=random_colormap) # plotting the polar coordinates as a scatter plot
    # Add colorbar for reference beside each point graph
    cbar = plt.colorbar(scatter, ax=ax[n], label='Magnitude Color Reference (Point %d)'% n, shrink=0.8)
    # ax.set_rmin(n, np.min(mags_3[0,n,:]))
    # ax.set_rmax(1, n, np.max(mags_3[0,n,:]) + 1)
    # ax.set_rlim(1, n, np.min(mags_3), np.max(mags_3) + 1)
# adding padding between graphs
plt.subplots_adjust(wspace = 1)
plt.savefig('plot_mags_aoas/straight_line_aoas+mags_per_point.png')

# Graph 4 - Generate plot with points at each end
mags_botleft = mags_fromloss[:,1]
mags_topright = mags_fromloss[:,40400]
mags_topleft = mags_fromloss[:,16198]  # this is a point that seems to be in the top left based on the ray angles
mags_botright = mags_fromloss[:,29808] # this is a point that seems to be in the bot right based on the ray angles

aoas_botleft = np.deg2rad(aoas[:,0,1]) # first set of rays, azimuth, first 10 rays on bottom left position
aoas_topright = np.deg2rad(aoas[:,0,40400]) # first set of rays, azimuth, first 10 on top right position
aoas_topleft = np.deg2rad(aoas[:,0,16198]) # this is a point that seems to be in the top left based on the ray angles
aoas_botright = np.deg2rad(aoas[:,0,29808]) # this is a point that seems to be in the bot right based on the ray angles


# Plot in polar coordinates, changing min and max of radial coordinates based on magnitude range
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
scatter_botleft = ax.scatter(aoas_botleft, mags_botleft, c=mags_botleft, cmap='Reds') # plotting the polar coordinates as a scatter plot
scatter_topright = ax.scatter(aoas_topright, mags_topright, c=mags_topright, cmap='Purples') # plotting the polar coordinates as a scatter plot
scatter_topleft = ax.scatter(aoas_topleft, mags_topleft, c=mags_topleft, cmap='Greens') # plotting the polar coordinates as a scatter plot
scatter_botright = ax.scatter(aoas_botright, mags_botright, c=mags_botright, cmap='Oranges') # plotting the polar coordinates as a scatter plot

ax.set_rmin(np.min(mags_fromloss))
ax.set_rmax(np.max(mags_fromloss) + 1)

# Add colorbar for reference - purple to yellow, purple is lower magnitude
cbar1 = plt.colorbar(scatter_botleft, ax=ax, label='Bottom Left Magnitudes')
cbar2 = plt.colorbar(scatter_topright, ax=ax, label='Top Right Magnitudes')
cbar3 = plt.colorbar(scatter_topleft, ax=ax, label='Top Left Magnitudes')
cbar4 = plt.colorbar(scatter_botright, ax=ax, label='Bottom Right Magnitudes')

plt.subplots_adjust(wspace = 1)

plt.savefig('plot_mags_aoas/10_rays_at_each_corner_fig_4.png')

##### Clustering #######
def convert_negative_to_positive(angles):
    # Convert angles to degrees
    angles_degrees = np.rad2deg(angles)
    # Convert negative angles to positive within the range [0, 360)
    converted_angles_degrees = np.where(angles_degrees < 0, 360 + angles_degrees, angles_degrees)
    # Convert back to radians
    converted_angles_radians = np.radians(converted_angles_degrees)
    return converted_angles_radians

# mags_sample = np.trim_zeros(mags_3[0, 1, :].reshape(-1)) #mags_botleft #mags_3[0, 1, :].reshape(-1) # # #  # Reshape to (100,)
# aoas_sample = np.trim_zeros(aoas_3[0, 1, :])#aoas_botleft #aoas_3[0, 1, :]# # # # Reshape to (100,)

# mags_sample = np.trim_zeros(mags_botleft) # # #  # Reshape to (100,)
# aoas_sample = np.trim_zeros(aoas_botleft) #aoas_3[0, 1, :]# # # # Reshape to (100,)

# mags_sample = np.trim_zeros(mags_3[0, 1, :].reshape(-1)) #mags_botleft #mags_3[0, 1, :].reshape(-1) # # #  # Reshape to (100,)
# aoas_sample = np.trim_zeros(aoas_3[0, 1, :])#aoas_botleft #aoas_3[0, 1, :]# # # # Reshape to (100,)

mags_sample = np.trim_zeros(mags_fromloss[:,7890]) # this is a point that seems to be in the bot right based on the ray angles
aoas_sample = np.trim_zeros(np.deg2rad(aoas[:,0,7890]))




# Function to wrap angles to the range [0, 360)

# Wrap angles
converted_aoas_angles = convert_negative_to_positive(aoas_sample)
#print(converted_aoas_angles[:20])

# Reshape the wrapped angles
data_aoas = converted_aoas_angles.reshape(-1, 1)

def kmeans_cluster_aoas(data_aoas,converted_aoas_angles,mag_samples):
    means_aoa = []
    inertias_aoa = []
    optimal_k_aoa = None
        
    for k in range(1, 10): 
        kmeans_aoa = KMeans(n_clusters=k)
        kmeans_aoa.fit(data_aoas)
        
        inertias_aoa.append(kmeans_aoa.inertia_)
        means_aoa.append(k)

        # Check if we've found the optimal k
        if k > 1 and inertias_aoa[-2] - inertias_aoa[-1] < 0.1 * (inertias_aoa[0] - inertias_aoa[-1]):
            optimal_k_aoa = k - 1  # Subtracting 1 to get the last value of k
            break

    # If optimal_k is still None, it means we haven't found the optimal k
    if optimal_k_aoa is None:
        # Alternatively, you could set optimal_k to the last value of k
        optimal_k_aoa = k

    print("Optimal k for just aoa:", optimal_k_aoa)

    # Plotting a separate plot using the k means clustering
    kmeans_aoa = KMeans(n_clusters=optimal_k_aoa)
    kmeans_aoa.fit(data_aoas)

    fig = plt.subplots(subplot_kw=dict(projection="polar"))
    plt.scatter(converted_aoas_angles, mag_samples, c=kmeans_aoa.labels_)
    plt.colorbar(label='Cluster Number')
    plt.title('Clustered Data, only AoAs')
    plt.xlabel('AoA')
    plt.ylabel('Magnitude')
    plt.savefig('kmeans-aoa-only.png')

    # Look at an elbow plot
    plt.figure(figsize=(10, 5))
    plt.plot(means_aoa, inertias_aoa, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig('kmeans-test-elbow-plot-aoa.png')



########
# Trying out Isolation Forest
######### 

# Define Isolation Forest model
isolation_forest = IsolationForest()

# Fit the model to the data
isolation_forest.fit(data_aoas)

# Predict outliers (-1) and inliers (1)
cluster_labels = isolation_forest.predict(data_aoas)

inlier_indicies = np.where(cluster_labels == 1)

data_aoaos_inliers = data_aoas[inlier_indicies]
aoas_inliers = converted_aoas_angles[inlier_indicies]
mags_inliers = mags_sample[inlier_indicies]

# Plot clustered data
fig = plt.subplots(subplot_kw=dict(projection="polar"))
plt.scatter(converted_aoas_angles, mags_sample, c=cluster_labels)
plt.title('Clustered Data, Outlier Clustering (Isolation Forest)')
plt.xlabel('AoAs')
plt.ylabel('Magnitudes')
plt.colorbar(label='Cluster Label')
plt.savefig('IsolationForest-test-plot.png')

# kmeans_cluster_aoas(data_aoaos_inliers, aoas_inliers, mags_inliers)

########
# Trying out DBSCAN clustering
######### 

# # Perform clustering with DBSCAN on the wrapped angles
dbscan_aoa = DBSCAN(eps=0.2, min_samples=5)
aoa_cluster_labels = dbscan_aoa.fit_predict(data_aoaos_inliers) # if any of the aoa clusters are negative then they aren't part of a group

# Plot clustered data
aoaos_inliers = converted_aoas_angles[inlier_indicies]
mags_inliers = mags_sample[inlier_indicies]
fig = plt.subplots(subplot_kw=dict(projection="polar"))
print(aoa_cluster_labels)
print(np.where(aoa_cluster_labels < 0))
plt.scatter(aoaos_inliers, mags_inliers, c=aoa_cluster_labels)
plt.title('Clustered Data, only AoAs (DBSCAN)')
plt.xlabel('AoAs')
plt.ylabel('Magnitudes')
plt.colorbar(label='Cluster Label')
plt.savefig('DBSCAN-test-plot.png')



########
# Trying out Local Neighbors
######### 

# # Define Local Outlier Factor model
# lof = LocalOutlierFactor()

# # Fit the model to the data and predict outlier scores
# # lower/negative outlier scores mean less relation, higher/positive scores mean closer relation (inlier)
# outlier_scores = lof.fit_predict(data_aoas)

# # Plot clustered data
# fig = plt.subplots(subplot_kw=dict(projection="polar"))
# plt.scatter(converted_aoas_angles, mags_sample, c=outlier_scores)
# plt.title('Clustered Data, Outlier Detection (Local Outlier Factor)')
# plt.xlabel('AoAs')
# plt.ylabel('Magnitudes')
# plt.colorbar(label='Outlier Score')
# plt.savefig('LocalNeighbors-test-plot.png')

# after identifying outliers, I can apply kMeans



########
# Trying out k-means clustering
########


# means = []
# means_both = []

# inertias = []
# inertias_both = []
# optimal_k_aoa = None
# optimal_k_both = None

# print(aoas_sample[:20])

# # Example usage:
# converted_aoas_angles = convert_negative_to_positive(aoas_sample)
# print(converted_aoas_angles[:20])

# data_aoas = converted_aoas_angles.reshape(-1, 1)
# data = np.stack((converted_aoas_angles, mags_sample), axis=1) 
# for k in range(1, 10): 
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(data_aoas)
    
#     inertias.append(kmeans.inertia_)
#     means.append(k)

#     # Check if we've found the optimal k
#     if k > 1 and inertias[-2] - inertias[-1] < 0.1 * (inertias[0] - inertias[-1]):
#         optimal_k_aoa = k - 1  # Subtracting 1 to get the last value of k
#         break
# for k in range(1, 10):   
#     k_means_mags_aoa = KMeans(n_clusters=k)
#     k_means_mags_aoa.fit(data)

#     inertias_both.append(k_means_mags_aoa.inertia_)

#     means_both.append(k)

#     # Check if we've found the optimal k
#     if k > 1 and inertias_both[-2] - inertias_both[-1] < 0.1 * (inertias_both[0] - inertias_both[-1]):
#         optimal_k_both = k - 1  # Subtracting 1 to get the last value of k
#         break

# # If optimal_k is still None, it means we haven't found the optimal k
# if optimal_k_aoa is None:
#     # Alternatively, you could set optimal_k to the last value of k
#     optimal_k_aoa = k
# if optimal_k_both is None:
#     optimal_k_both = k

# print("Optimal k for just aoa:", optimal_k_aoa)
# print("Optimal k for both mags and aoa:", optimal_k_both)

# # Plotting a seperated plot using the k means clustering

# kmeans_aoa = KMeans(n_clusters=optimal_k_aoa)
# kmeans_aoa.fit(data_aoas)

# k_means_mags_aoa = KMeans(n_clusters=optimal_k_both)
# k_means_mags_aoa.fit(data)

# fig, axs = plt.subplots(3, subplot_kw=dict(projection="polar"), figsize=(12, 12))
# fig.suptitle('Comparison of Clustering Results with KMeans (k_both = {}, k_aoa = {})'.format(optimal_k_both, optimal_k_aoa))

# # Plot original data
# axs[0].scatter(converted_aoas_angles, mags_sample)
# axs[0].set_title('Original Data')

# # Plot clustered data, kmeans with both aoa and mags data
# axs[1].scatter(converted_aoas_angles, mags_sample, c=k_means_mags_aoa.labels_)
# axs[1].set_title('Clustered Data, Mags and AoAs')

# # Plot clustered data
# axs[2].scatter(converted_aoas_angles, mags_sample, c=kmeans_aoa.labels_)
# axs[2].set_title('Clustered Data, only AoAs')

# plt.subplots_adjust(hspace = 0.4)
# plt.savefig('kmeans-comparison.png')

# # # look at an elbow plot
# fig = plt.subplots(figsize=(10,5))
# plt.plot(means, inertias,'o-')
# plt.plot(means_both, inertias_both,'o-')

# plt.xlabel('Number of Clusters')
# plt.ylabel('Inertia')
# plt.grid(True)
# plt.savefig('kmeans-test-elbow-plot.png')
