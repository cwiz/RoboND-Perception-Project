## Project: 3D Perception

**Sergei Surovtsev**
<br/>
Udacity Robotics Software Engineer Nanodegree
<br/>
Class of November 2018

## Project Description

This project is an introduction to 3D perception in robotics. It involves RGBZ camera, point clowns, and python tools for processing point clouds. It also covers basic segmentation and object recognition techniques.

The problem we are solving in this project is identifying objects located on a table and then moving these objects to appropriate bits. Skills obtained from completing this projects can be applied to assembly line robotics. 

Robotic arm used in this project is [Willow Garage PR2](https://www.youtube.com/watch?v=cue7EHeY3i4).

## Project Goals

* Introduction to 3D perception using RGBZ cameras
* Intruduction to basic algorithms for clustering and segmentation

## Technical Formulation of Problem 

* Set up environment as described in [Project Repository](https://github.com/udacity/RoboND-Perception-Project)
* Complete Labs in Udacity lectures describing perception pipeline
* Port code from labs to a project and tune hyperparameters
* Perform object recognition for 3 test scenes
* [optional] Perform constains analysis and complete kinematics phase

## Mathematical Models

### Object Recognition

First step towards object recognition is training an SVM model. We're extracting each object features and then build 96 histogram as an input vector. Here's code doing that:

```python
	def compute_color_histograms(cloud, using_hsv=True):

	    # Compute histograms for the clusters
	    point_colors_list = []

	    # Step through each point in the point cloud
	    for point in pc2.read_points(cloud, skip_nans=True):
	        rgb_list = float_to_rgb(point[3])
	        if using_hsv:
	            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
	        else:
	            point_colors_list.append(rgb_list)

	    # Populate lists with color values
	    channel_1_vals = []
	    channel_2_vals = []
	    channel_3_vals = []

	    for color in point_colors_list:
	        channel_1_vals.append(color[0])
	        channel_2_vals.append(color[1])
	        channel_3_vals.append(color[2])
	    
	    # 1. Compute histograms
	    hist_1, _ = np.histogram(channel_1_vals, bins=96, density=True, range=(0.0, 255.0))
	    hist_2, _ = np.histogram(channel_2_vals, bins=96, density=True, range=(0.0, 255.0))
	    hist_3, _ = np.histogram(channel_3_vals, bins=96, density=True, range=(0.0, 255.0))

	    # 2. Concatenate and normalize the histograms
	    features = np.concatenate((hist_1, hist_2, hist_3), axis=0) 

	    return features #/ np.sum(features) 


	def compute_normal_histograms(normal_cloud):
	    norm_x_vals = []
	    norm_y_vals = []
	    norm_z_vals = []

	    for norm_component in pc2.read_points(normal_cloud,
	                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
	                                          skip_nans=True):
	        norm_x_vals.append(norm_component[0])
	        norm_y_vals.append(norm_component[1])
	        norm_z_vals.append(norm_component[2])

	    # TODO: Compute histograms of normal values (just like with color)
	    hist_1, _ = np.histogram(norm_x_vals, bins=96, density=True, range=(-np.pi/2, np.pi/2))
	    hist_2, _ = np.histogram(norm_y_vals, bins=96, density=True, range=(-np.pi/2, np.pi/2))
	    hist_3, _ = np.histogram(norm_z_vals, bins=96, density=True, range=(-np.pi/2, np.pi/2))

	    # TODO: Concatenate and normalize the histograms
	    features = np.concatenate((hist_1, hist_2, hist_3), axis=0) 

	    return features #/ np.sum(features)
```

Then we use trained model for object recognition. Here's complete pipeline:

```python
# 0. Convert ROS msg to PCL data
cloud_filtered = ros_to_pcl(pcl_msg)

# Much like the previous filters, we start by creating a filter object: 
outlier_filter = cloud_filtered.make_statistical_outlier_filter()

# Set the number of neighboring points to analyze for any given point
outlier_filter.set_mean_k(3)

# Set threshold scale factor
x = 0.00001

# Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
outlier_filter.set_std_dev_mul_thresh(x)

# Finally call the filter function for magic
cloud_filtered = outlier_filter.filter()

# 1. Voxel Grid filter
vox = cloud_filtered.make_voxel_grid_filter()

LEAF_SIZE = 0.005   
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

# Call the filter function to obtain the resultant downsampled point cloud
cloud_filtered = vox.filter()

# 2. PassThrough filter
# Z direction
passthrough_1 = cloud_filtered.make_passthrough_filter()

filter_axis_1 = 'z'
passthrough_1.set_filter_field_name(filter_axis_1)
axis_1_min = 0.6
axis_1_max = 1.1
passthrough_1.set_filter_limits(axis_1_min, axis_1_max)
cloud_filtered = passthrough_1.filter()

# X direction
passthrough_2 = cloud_filtered.make_passthrough_filter()

filter_axis_2 = 'x'
passthrough_2.set_filter_field_name(filter_axis_2)
axis_2_min = 0.34
axis_2_max = 1.0
passthrough_2.set_filter_limits(axis_2_min, axis_2_max)
cloud_filtered = passthrough_2.filter()

# 3. RANSAC plane segmentation

# Create the segmentation object
seg = cloud_filtered.make_segmenter()

# Set the model you wish to fit 
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)

# Max distance for a point to be considered fitting the model
# Experiment with different values for max_distance 
# for segmenting the table
max_distance = 0.01
seg.set_distance_threshold(max_distance)

# Call the segment function to obtain set of inlier indices and model coefficients
inliers, coefficients = seg.segment()

# 4. Extract inliers
extracted_inliers = cloud_filtered.extract(inliers, negative=False)

# 5. Extract outliers
extracted_outliers = cloud_filtered.extract(inliers, negative=True)

#6. Euclidean Clustering
white_cloud = XYZRGB_to_XYZ(extracted_outliers)
tree = white_cloud.make_kdtree()
# Create a cluster extraction object
ec = white_cloud.make_EuclideanClusterExtraction()
# Set tolerances for distance threshold 
# as well as minimum and maximum cluster size (in points)
# NOTE: These are poor choices of clustering parameters
# Your task is to experiment and find values that work for segmenting objects.
ec.set_ClusterTolerance(0.02)
ec.set_MinClusterSize(40)
ec.set_MaxClusterSize(9000)
# Search the k-d tree for clusters
ec.set_SearchMethod(tree)
# Extract indices for each of the discovered clusters
cluster_indices = ec.Extract()


# 7. Create Cluster-Mask Point Cloud to visualize each cluster separately
#Assign a color corresponding to each segmented object in scene
cluster_color = get_color_list(len(cluster_indices))

color_cluster_point_list = []

for j, indices in enumerate(cluster_indices):
    for i, indice in enumerate(indices):
        color_cluster_point_list.append([white_cloud[indice][0],
                                        white_cloud[indice][1],
                                        white_cloud[indice][2],
                                         rgb_to_float(cluster_color[j])])

#Create new cloud containing all clusters, each with unique color
cluster_cloud = pcl.PointCloud_PointXYZRGB()
cluster_cloud.from_list(color_cluster_point_list)


# 8. Convert PCL data to ROS messages
ros_pcl_table   = pcl_to_ros(extracted_inliers)
ros_pcl_objects = pcl_to_ros(extracted_outliers)
ros_pcl_cluster = pcl_to_ros(cluster_cloud)

# 9. Publish ROS msg
pcl_objects_pub.publish(ros_pcl_objects)
pcl_table_pub.publish(ros_pcl_table)
pcl_cluster_pub.publish(ros_pcl_cluster)

# 10. 

detected_objects_labels = []
detected_objects = []

for index, pts_list in enumerate(cluster_indices):
    # Grab the points for the cluster from the extracted outliers (cloud_objects)
    pcl_cluster = extracted_outliers.extract(pts_list)
    # convert the cluster from pcl to ROS using helper function
    ros_cluster = pcl_to_ros(pcl_cluster)

    # Extract histogram features
    chists = compute_color_histograms(ros_cluster, using_hsv=True)
    normals = get_normals(ros_cluster)
    nhists = compute_normal_histograms(normals)
    feature = np.concatenate((chists, nhists))
    # labeled_features.append([feature, model_name])

    # Make the prediction, retrieve the label for the result
    # and add it to detected_objects_labels list
    prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
    label = encoder.inverse_transform(prediction)[0]
    detected_objects_labels.append(label)

    # Publish a label into RViz
    label_pos = list(white_cloud[pts_list[0]])
    label_pos[2] += .2
    object_markers_pub.publish(make_label(label,label_pos, index))

    # Add the detected object to the list of detected objects.
    do = DetectedObject()
    do.label = label
    do.cloud = ros_cluster
    detected_objects.append(do)
```

### Results

KR210 is able to perform a task at hand. Kinematics part is wacky because no collision analysis is performed. [video](https://youtu.be/q34VwS6K64U)