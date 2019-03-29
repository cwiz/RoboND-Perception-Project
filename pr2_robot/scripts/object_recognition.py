#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

scene_number = 1
TEST_SCENE_NUM = Int32()
TEST_SCENE_NUM.data = scene_number

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

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
        chists = compute_color_histograms(ros_cluster, using_hsv=False)
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

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    # ----------------------------------------------------------------------------------
    # Invoke pr2_mover() function
    # ----------------------------------------------------------------------------------
    if len(detected_objects)>0:
        try:
            pr2_mover(detected_objects)
        except rospy.ROSInterruptException:
            pass
    else:
        rospy.logwarn('No detected objects !!!')
    
    return

def get_place_pose_for_group(group_name):
    dropbox = rospy.get_param('/dropbox')
    
    for group in dropbox:
        if group['group'] == group_name:           
            return tuple_to_ros_point(group['position'])

def get_arm_name(group_name):
    arm_name = String()
    if group_name == "red":
        arm_name.data = "left"
    else:
        arm_name.data = "right"
    return arm_name

def tuple_to_ros_point(tpl):
    pick_pose = Pose()
    pick_pose.position.x = tpl[0]
    pick_pose.position.y = tpl[1]
    pick_pose.position.z = tpl[2]

    return pick_pose

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    object_list_param = rospy.get_param('/object_list')
    output_items = []
    OBJECT_NAMES = []
    WHICH_ARMS   = []
    PICK_POSES   = []
    PLACE_POSES  = []

    # Parse parameters into individual variables
    for i in range(len(object_list_param)):

        name = object_list_param[i]['name']
        group = object_list_param[i]['group']
        centroid = np.mean(ros_to_pcl(object_list[i].cloud).to_array(), axis=0)[:3]
        centroid = [
            np.asscalar(centroid[0]),
            np.asscalar(centroid[1]),
            np.asscalar(centroid[2]),
        ]

        # Name
        OBJECT_NAME = String()
        OBJECT_NAME.data = name
        # Place Pose
        PLACE_POSE = get_place_pose_for_group(group)
        # Pick Post 
        PICK_POSE = tuple_to_ros_point(centroid)
        # Arm Name
        WHICH_ARM = get_arm_name(object_list_param[i]['group'])

        yaml_item = make_yaml_dict(
            TEST_SCENE_NUM,
            WHICH_ARM,
            OBJECT_NAME,
            PICK_POSE,
            PLACE_POSE,
        )

        WHICH_ARMS.append(WHICH_ARM)
        OBJECT_NAMES.append(OBJECT_NAME)
        PICK_POSES.append(PICK_POSE)
        PLACE_POSES.append(PLACE_POSE)


        output_items.append(yaml_item)

    print(output_items)
    send_to_yaml("output_1.yaml", output_items)
    rospy.wait_for_service('pick_place_routine')

    for i in range(len(object_list_param)):
        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(
                TEST_SCENE_NUM, 
                OBJECT_NAMES[i], 
                WHICH_ARMS[i], 
                PICK_POSES[i], 
                PLACE_POSES[i],
            )

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e


if __name__ == '__main__':

    rospy.init_node('object_recognition', anonymous=True)

    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluser", PointCloud2, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)

    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    while not rospy.is_shutdown():
        rospy.spin()
