"""
Contains functions for curved path implementations. Currently using
dataset_consumer.py for all implementations. 
"""
import h5py
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os
from typing import Callable


def breesenham(x0, y0, x1, y1):
    """
    Generate a line between two points using Breesenham's algorithm
    Returns: a list of points (x, y) to create a straight line 
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    points = []
    x_coor = []
    y_coor = []
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            x_coor.append(x)
            y_coor.append(y)
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            x_coor.append(x)
            y_coor.append(y)
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return ([points, x_coor, y_coor])

def curveInRange(delta, s, theta,x, y, range_of_points, L,dt):
    """
    Generate a curved line between two points using a method similar to car kinematics
    Inputs: 
        delta - steering angle (radians)
        s     - speed
        theta - car heading (radians)
        x     - x coordinates of start and end points
        y     - y coordinates of start and end points
        range - the number of points we want to have
        L     - Length of car (omitting this?)
        dt    - time increment   
    Returns: 2 lists, one with the (x, y) points to create the line, 
        and another with the heading angles 
    """
    xvec = []
    yvec = []
    points = []
    thetavec = []
    for i in range(len(range_of_points)):
        dx = np.cos(theta) * s * dt
        dy = np.sin(theta)* s *dt
        dtheta = (s/L) * np.tan(delta) * dt
        xnew = x + dx
        ynew = y + dy
        thetanew = theta + dtheta
        thetanew = np.mod(thetanew,np.pi) # Wrap theta at pi
        xvec.append(xnew)
        yvec.append(ynew)
        points.append((xnew, ynew))
        thetavec.append(thetanew)
        x = xnew
        y= ynew
        theta = thetanew
    # print(f"Generated Points: {points}")
    # print(f"Generated Angles: {thetavec}")

    return([xvec, yvec, points, thetavec]) 

########
# A class to process positon and CSI (magnitude/phase) data. 
# It initalizes the dataset and has methods to setup, process, generate, and display data
# functions:
    # scale: scales the data using a scalar function
    # unwrap: unwraps a given array
    # print_info: converts the object to json string
    # generate_straight_paths: continually finds points on the dataset grid until the number of 
        # points is greater than or equal to the desired number of paths. 
    # generate_curve_paths: uses a point and a steering angle to create a curved path and 
        # and find points on the dataset grid
    # Generate torch datasets with the following functions: 
        # paths_to_dataset_mag_only
        # paths_to_dataset_phase_only
        # paths_to_dataset_positions_only
        # pats_to_dataset_interleaved: dataset for both magnitude and phases
########
class DatasetConsumer:
    def __init__(self, dataset_path):
        self.attributes = None
        self.csi_mags = None
        self.csi_phases = None
        self.rx_positions = None

        with h5py.File(dataset_path, 'r') as file:
            self.attributes = self.__clean_attributes(file.attrs)
            self.csi_mags = file['csis_mag'][:]
            self.csi_phases = file['csis_phase'][:]
            self.rx_positions = file['positions'][:]

        self.tx_position = self.attributes['tx_position']
        self.grid_size, self.grid_spacing = self.__find_grid(self.rx_positions)

    def __find_grid(self, rx_positions):
        # Find the grid size and spacing that was used to generate the dataset
        min_x = np.min(rx_positions[0, :])
        max_x = np.max(rx_positions[0, :])
        min_y = np.min(rx_positions[1, :])
        max_y = np.max(rx_positions[1, :])
        grid_bounds = (min_x, max_x, min_y, max_y)

        grid_spacing = self.attributes['rx_grid_spacing']

        return grid_bounds, grid_spacing
    

    def __real_to_grid(self, x, y):
        # Find the index in the grid that the given point is in
        # Return None if the point is not in the grid
        if x < self.grid_size[0] or x > self.grid_size[1]:
            return None
        if y < self.grid_size[2] or y > self.grid_size[3]:
            return None

        x_index = int((x - self.grid_size[0]) / self.grid_spacing)
        y_index = int((y - self.grid_size[2]) / self.grid_spacing)

        return x_index, y_index

    def __grid_to_real_index(self, x, y):
        # Find the real position of the given grid index
        x_real = self.grid_size[0] + (x * self.grid_spacing)
        y_real = self.grid_size[2] + (y * self.grid_spacing)

        # Find the closest point in the rx_positions array
        index = np.argmin(np.abs(self.rx_positions[0, :] - x_real) + np.abs(self.rx_positions[1, :] - y_real))
        return index

    def __clean_attributes(self, attributes):
        def replace_np_array_with_list(d):
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    if v.size == 1:
                        d[k] = v.item()
                    else:
                        d[k] = v.tolist()
                elif isinstance(v, dict):
                    replace_np_array_with_list(v)

        d = dict(attributes)
        replace_np_array_with_list(d)
        return d
    
    # Scale data with passed function
    def scale(self, scaler: Callable[[np.array], np.array], data: np.array) -> np.array:
        return scaler(data)
    
    # Unwrap data with passed function
    def unwrap(self, data: np.array, axis:int=0) -> np.array:
        return np.unwrap(data, axis=axis)

    def print_info(self):
        print(json.dumps(self.attributes, indent=2))
        

    def generate_straight_paths(self, num_paths, path_length_n=5):
        """
        Generate straight paths in the rx_positions array.

        num_paths: Number of paths to generate
        path_length_n: Length of each path in number of points
        """
        print(f'Generating {num_paths} paths of length {path_length_n}')
        
        path_indices = np.zeros((num_paths, path_length_n), dtype=np.int32)
        for i in range(num_paths):
            while True:
                # Grab two random points
                point1 = np.random.randint(0, self.rx_positions.shape[1])
                point2 = np.random.randint(0, self.rx_positions.shape[1])

                # Generate a straight line between the two points
                x0, y0 = self.__real_to_grid(self.rx_positions[0, point1], self.rx_positions[1, point1])
                x1, y1 = self.__real_to_grid(self.rx_positions[0, point2], self.rx_positions[1, point2])
                [points, x_coor, y_coor] = breesenham(x0, y0, x1, y1)

                # Make sure the path is at least path_length_n long
                if len(points) < path_length_n:
                    continue
                path_indices[i] = [self.__grid_to_real_index(x, y) for x, y in points[:path_length_n]]
                break
        # plt.plot(x_coor, y_coor, "green")
        # plt.show()
        return path_indices
    
    def generate_curved_paths(self, num_paths, path_length_n=20):
        """
        Generate curved paths in the rx_positions array. (currently not working as expected)

        inputs:
            num_paths: Number of paths to generate
            path_length_n: Length of each path in number of points
        output: An array with shape: (num_paths, path_length_n)

        """
        deg2rad = np.pi / 180.0
        rad2deg = 180.0 / np.pi 
        print(f'Generating {num_paths} paths of length {path_length_n}')
        path_indices = np.zeros((num_paths, path_length_n), dtype=np.int32)
        for i in range(num_paths):
            while True:
                # Grab one random point within the size of the dataset 
                point1 = np.random.randint(0, self.rx_positions.shape[1])
                print(point1)
                # reducing randomness by setting 2 degrees
                steering_angle_delta = np.random.uniform(-0.5, 0.5)
                heading_angle_theta = np.random.uniform(0, 360)

                # angles in radians
                delta = steering_angle_delta * deg2rad
                theta = heading_angle_theta * deg2rad
                dt = 1 # this value can be changed, currently the time step is 1

                # Generate a curved line between from random point and set angle
                x, y = self.__real_to_grid(self.rx_positions[0, point1], self.rx_positions[1, point1])
                range_of_points = np.arange(0,path_length_n,dt)
                [x_coor, y_coor, points, thetas] = curveInRange(delta, 1, theta, x, y, range_of_points, 0.2, dt)
                
                # Check if all the points are within the grid bounds
                ep = 10
                if np.any(np.array(x_coor) < self.grid_size[0] + ep) or np.any(np.array(x_coor) > self.grid_size[1] - ep):
                    continue
                if np.any(np.array(y_coor) < self.grid_size[2] + ep) or np.any(np.array(y_coor) > self.grid_size[3] - ep):
                    continue

                # Make sure the path is at least path_length_n long
                path_indices[i] = [self.__grid_to_real_index(x, y) for x, y in points[:path_length_n]]
                break

        # plt.plot(x_coor, y_coor,  "blue")
        # plt.show()
        return path_indices
    
    def generate_left_right_curves(self, num_paths, path_length_n=20, heading_angle_theta=0):
        """
        Generate curved paths in the rx_positions array. (currently not working as expected)

        inputs:
            num_paths: Number of paths to generate
            path_length_n: Length of each path in number of points
        output: An array with shape: (num_paths, path_length_n)

        """
        # generate a curved path to the left, and to the right (so pos and neg steering angle)
        deg2rad = np.pi / 180.0
        rad2deg = 180.0 / np.pi 
        print(f'Generating {num_paths} paths of length {path_length_n}')
        path_indices_1 = np.zeros((num_paths, path_length_n), dtype=np.int32)
        path_indices_2 = np.zeros((num_paths, path_length_n), dtype=np.int32)
        for i in range(num_paths):
            while True:
                # Grab one random point within the size of the dataset 
                point1 = np.random.randint(0, self.rx_positions.shape[1])
                print(point1)
                # reducing randomness by setting 2 degrees
                steering_angle_delta = np.random.uniform(-0.5, 0.5)
                heading_angle_theta = np.random.uniform(0, 360)

                # angles in radians
                delta = steering_angle_delta * deg2rad
                theta = heading_angle_theta * deg2rad
                dt = 1 # this value can be changed, currently the time step is 1

                # Generate a curved line between from random point and set angle
                x, y = self.__real_to_grid(self.rx_positions[0, point1], self.rx_positions[1, point1])
                range_of_points = np.arange(0,path_length_n,dt)
                [x_coor, y_coor, points_curve1, thetas] = curveInRange(delta, 1, theta, x, y, range_of_points, 0.2, dt)
                [x_coor, y_coor, points_curve2, thetas] = curveInRange(-1*delta, 1, theta, x, y, range_of_points, 0.2, dt)
               
                # Check if all the points are within the grid bounds
                ep = 10
                if np.any(np.array(x_coor) < self.grid_size[0] + ep) or np.any(np.array(x_coor) > self.grid_size[1] - ep):
                    continue
                if np.any(np.array(y_coor) < self.grid_size[2] + ep) or np.any(np.array(y_coor) > self.grid_size[3] - ep):
                    continue

                # Make sure the path is at least path_length_n long
                path_indices_1[i] = [self.__grid_to_real_index(x, y) for x, y in points_curve1[:path_length_n]]
                path_indices_2[i] = [self.__grid_to_real_index(x, y) for x, y in points_curve2[:path_length_n]]
                break

        # plt.plot(x_coor, y_coor,  "blue")
        # plt.show()
        return path_indices_1, path_indices_2
    
    def paths_to_dataset_mag_only(self, path_indices):
        """
        Generate a torch dataset from the given path indices
        Shape: (num_paths, path_length_n, 128)
        """
        # Use the indices to grab the CSI data for each point
        csi_mags = self.csi_mags[:, path_indices]
        csi_mags = np.swapaxes(csi_mags, 0, 1)
        csi_mags = np.swapaxes(csi_mags, 1, 2)
        return csi_mags
    
    def paths_to_dataset_phase_only(self, path_indices):
        """
        Generate a torch dataset from the given path indices
        Shape: (num_paths, path_length_n, 128)
        """
        # Use the indices to grab the CSI data for each point
        csi_phases = self.csi_phases[:, path_indices]
        csi_phases = np.swapaxes(csi_phases, 0, 1)
        csi_phases = np.swapaxes(csi_phases, 1, 2)
        return csi_phases
    
    def paths_to_dataset_positions(self, path_indices):
        """
        Generate a torch dataset from the given path indices
        Shape: (num_paths, path_length_n, 2)
        """
        # Use the indices to grab the positions for each point
        positions = self.rx_positions[:, path_indices]
        positions = np.swapaxes(positions, 0, 1)
        return positions
    
    def paths_to_dataset_interleaved(self, path_indices):
        """
        Generate a torch dataset from the given path indices
        Shape: (num_paths, path_length_n, 256)
        """
        # Get the magnitude and phase data
        csi_mags = self.paths_to_dataset_mag_only(path_indices)
        csi_phases = self.paths_to_dataset_phase_only(path_indices)

        # Create a new array to hold the interleaved data
        num_paths, path_length_n, _ = csi_mags.shape
        interleaved = np.empty((num_paths, path_length_n, 256), dtype=csi_mags.dtype)

        # Fill the new array with alternating slices from the two original arrays
        interleaved[..., ::2] = csi_mags
        interleaved[..., 1::2] = csi_phases

        return interleaved
        
# PLOTTING RESULTS TO CHECK
d = DatasetConsumer('./machine_learning/data/dataset_0_5m_spacing.h5')
# d = DatasetConsumer('older_ver/dataset.h5')

straight_paths = d.generate_straight_paths(3)
path_interleaved = d.paths_to_dataset_interleaved(straight_paths)
print(path_interleaved.shape) # should be 1x256, but for every point. So 1x20x256
print(path_interleaved) 
#paths_curved = d.generate_curved_paths(70)
#curve_pos = d.paths_to_dataset_positions(paths_curved)

# paths_curved_1, paths_curved_2 = d.generate_left_right_curves(10)
# curve_1_pos = d.paths_to_dataset_positions(paths_curved_1)
# curve_2_pos = d.paths_to_dataset_positions(paths_curved_2)

# # print(pathsC2.shape)
# # print(curve_pos.shape)

# plt.title("Positions")

# for i in range(10):
#     plt.plot(curve_1_pos[i, 0, :], curve_1_pos[i, 1, :],color="blue")
#     plt.plot(curve_2_pos[i, 0, :], curve_2_pos[i, 1, :],color="green")
# plt.show()

# for i in range(10):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(curve_pos[i,0,:], curve_pos[i,1,:], curve_pos[i,2,:])
#     ax.set_xlabel('$X$')
#     ax.set_ylabel('$Y$')
#     ax.set_zlabel('$Z$')
# plt.show()
print("here: ")
