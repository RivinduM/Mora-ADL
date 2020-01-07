<meta name="google-site-verification" content="R3PkXCRo_62TiBDDPFhhAJAD85UIXMd1Abenk3FoeKo" />

# Mora-ADL
A multi-modal Human Activities of Daily Living dataset and Data Collection Tool.


![Mora ADL](https://img.shields.io/badge/Mora--ADL-Dataset-green) ![Mora ADL-Tool](https://img.shields.io/badge/Mora--ADL-Data%20Collection%20Tool-orange) ![License: CC0](https://img.shields.io/github/license/RivinduM/Mora-ADL)  [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)


Table of contents
=================

<!--ts-->
   * [Dataset](#dataset)
      * [Dataset properties](#dataset-properties)
      * [Playing depth video](#playing-depth-video)
      * [Skeleton data stream](#skeleton-data-stream)
   * [Data Collection Tool](#data-collection-tool)
      * [Using the tool](#using-the-tool)
        * [Setting up dependencies](#setting-up-dependencies)
        * [Running the tool](#running-the-tool)
   * [Licence](#license)
   * [Citaton](#citation)
   * [Collaborators](#collaborators)
<!--te-->


# Dataset

## Link
[Mora-ADL Dataset](https://drive.google.com/drive/folders/1xKcP2oYMxIxbH7L-qGRABK9Xz5N3QdAA?usp=sharing)

## Dataset properties

7 data streams:
- Depth image stream
-  RGB image stream
-  Skeleton data stream
-  Silhouette stream
-  3 audio streams

24 Activities of Daily Living:
- Making a phone call, Clapping, Drinking, Eating, Entering from door, Exiting from door, Falling, Lying down, Opening pill container, Picking object, Reading, Sit still, Sitting down, Sleeping, Standing up, Sweeping, Using laptop, Using phone, Wake up, Walking, Washing hand, Watching TV, Water pouring and Writing

17 subjects:
- 11 males
- 6 females

Continuous dataset to test continuous activity classification.
- Performed by 3 subjects
- In 2 different environments

## Playing depth video

Run [playDepthVideo.py](https://github.com/RivinduM/Mora-ADL/blob/master/playDepthVideo.py) with the following arguments to play a depth video file.
- ```-d``` depth file path
- ```-s``` skeleton file path
- ```-sil``` (Optional) True/False [True to view silhouette view. False to view depth image view]

## Skeleton data stream

The skeleton data stream has 5 values per row and for each frame it has 15 such rows. The 15 rows gives the positions of the 15 joint positions as in the below figure.

![Skeleton joint position](https://github.com/RivinduM/Mora-ADL/blob/master/skeletonJoints.png "Skeleton joint positions")


The 5 values of the row stands for:
- X coordinate using the "real world" coordinate system
- Y coordinate using the "real world" coordinate system
- Z coordinate using the "real world" coordinate system
- X coordinates of the depth map
- Y coordinates of the depth map

# Data Collection Tool

Data collection tool written using python to collect depth, RGB, audio, silhouette and skeleton data using the Microsoft Kinect device and the OpenNI/NiTE tool. 

## Using the tool

### Setting up dependencies

Please follow the following steps if you have not installed Kinect SDK, OpenNI or NiTE tool.

#### Step 1
1. Download & install Kinect SDK 1.8 or higher version from [here](https://www.microsoft.com/en-us/download/details.aspx?id=40278)  or another source.
2. Download & install OpenNI 2.2 or higher version from [here](https://structure.io/openni) or another source.
3. To verify the setup run SimpleViewer in *Program files -> OpenNI2 -> Samples -> Bin*
4. Download & install Nite 2.2 or higher version from [here](https://drive.google.com/file/d/0B3e4_6C5_YOjOGIySEluYkNibEE/edit) or another source.
5. To verify the setup run UserViewer in *Program files -> PrimeSense -> NiTE2 ->Samples -> Bin*

#### Step 2
1. Download openni-python repository from [here](https://github.com/severin-lemaignan/openni-python).
2. Extract it to a preferred location.
3. Copy NiTE2 and OpenNI2 folders in *Program files -> PrimeSense -> NiTE2 -> Samples* to the openni-python.

#### Step 3
1. Copy the dataset tool ([dataCollectionTool.py](https://github.com/RivinduM/Mora-ADL/blob/master/dataCollectionTool.py)) and [audioDevices.py](https://github.com/RivinduM/Mora-ADL/blob/master/audioDevices.py) to the openni-python folder.

### Running the tool

First, run the [audioDevices.py](https://github.com/RivinduM/Mora-ADL/blob/master/audioDevices.py) to detect the port numbers of the microphones.

Then run [dataCollectionTool.py](https://github.com/RivinduM/Mora-ADL/blob/master/dataCollectionTool.py) in the command line using the following command.
```sh
$ python audioDevices.py -d <<microphone_ports>> -p <<location_to_save_data>> -s <<subject_name>> -a <<act_name>> 
```

To stop recording press ```Ctrl + c``` in the command line.



License
----

Creative Commons Zero v1.0 Universal

Citation
----
If you use the Mora-ADL dataset or the Mora-ADL Data Collection Tool please cite:
>@article{To be updated soon
}

Collaborators
----
Rivindu Madushan

Danushkka Madhuranga

Chathuranga Siriwardhana
