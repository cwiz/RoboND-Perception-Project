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

#### Feature Extraction

First step towards object recognition is training an SVM model. 

In this case we use histograms over RGB channels and hisotgrams over surface normals. 
Conversion to HSV workd fine during training but real results were faulty. My speculation that in our case color information of objects is more important than color intensity values.

Code used for feature extraction: 

Confusion matrix we get from extracting features:

![segmentation-obstacles](https://github.com/cwiz/RoboND-Perception-Project/blob/master/images/figure_2.png?raw=true "Confusion Matrix")

Then we use trained model for object recognition. [link](https://github.com/cwiz/RoboND-Perception-Project/blob/master/pr2_robot/scripts/features.py) 

#### Processing Pipeline

1. Statistical outlier filtering
2. Voxel grid downsampling
3. Passthough filter over z and x directions
4. RANSAC plane segmentation, object extraction
5. Object clustering using euclidean distance
6. Object recognition using previously trained SVM model

[link](https://github.com/cwiz/RoboND-Perception-Project/blob/master/pr2_robot/scripts/object_recognition.py) 

### Results

KR210 is able to perform a task at hand. Kinematics part is wacky because no collision analysis is performed. [video](https://youtu.be/q34VwS6K64U)

#### Scenario 1 

![scenario_1](https://github.com/cwiz/RoboND-Perception-Project/blob/master/results/output_1.png?raw=true "Scenario 1 Output")

[yaml output](https://github.com/cwiz/RoboND-Perception-Project/blob/master/results/output_1.yaml?raw=true)

#### Scenario 2

![scenario_2](https://github.com/cwiz/RoboND-Perception-Project/blob/master/results/output_2.png?raw=true "Scenario 2 Output")

[yaml output](https://github.com/cwiz/RoboND-Perception-Project/blob/master/results/output_2.yaml?raw=true)

#### Scenario 3

![scenario_2](https://github.com/cwiz/RoboND-Perception-Project/blob/master/results/output_3.png?raw=true "Scenario 2 Output")

[yaml output](https://github.com/cwiz/RoboND-Perception-Project/blob/master/results/output_3.yaml?raw=true)
