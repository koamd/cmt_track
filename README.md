# Introduction

CMT (Consensus-based Matching and Tracking of Keypoints for Object Tracking) is a novel keypoint-based method for long-term model-free object tracking in a combined matching-and-tracking framework. Details can be found on the [project page](http://www.gnebehay.com/cmt)
and in their [publication](http://www.gnebehay.com/publications/wacv_2014/wacv_2014.pdf).
The Python implementation in this repository is platform-independent and runs on Linux, Windows and OS X.

#License
CMT is freely available under the [3-clause BSD license][1],
meaning that you can basically do with the code whatever you want. If you use our algorithm in scientific work, please cite their publication

```
@inproceedings{Nebehay2014WACV,
    author = {Nebehay, Georg and Pflugfelder, Roman},
    booktitle = {Winter Conference on Applications of Computer Vision},
    month = mar,
    publisher = {IEEE},
    title = {Consensus-based Matching and Tracking of Keypoints for Object Tracking},
    year = {2014}
}
```

# Package Dependencies

Create your own virtual environment

```
python -m venv venv
```

Activate virtual environment

```
. venv/bin/activate
```

Install the required packages in your own pip environment.

```pip
pip install -r requirements.txt
```

## How to use CMT Multi Object Tracker (CMT MOT)

Inside [CMT.py](CMT.py), there is a MultiCMT Class Object that can concurrently process multiple CMT Tracks

To Initialize a CMT MOT, 

```python
import CMT

CMT_multi_tracker = CMT.MultiCMT()
```

To add a detection track, 

```python
#im_gray0 is a opencv grayscale image
#expandedBoxList is a list of list. Format is in [[tl_x, tl_y, br_x, br_y], [tl_x, tl_y, br_x, br_y], ...]
CMT_multi_tracker.append_detections(im_gray0, expandedBoxList)
```

An optional label list can be provided to the multi_tracker

```python
#the labels are meant to give each tracker a unique track_id. Format is a list of string. E.g ['object_1','object_2','object_3',....]) 
#the length of the labels must be the same as the number of bounding boxes supplied
CMT_multi_tracker.append_detections(im_gray0, expandedBoxList, labels)
```



If there are no new detections, we can use the following function to perform predictions of tracker positions in the current frame based on information from previous frames.

```python
CMT_multi_tracker.process_tracks(im_gray0)

```

To access the list of detections:

```python
#to get list of detections in [[tl_x, tl_y, w, h], [tl_x, tl_y, w, h], ... ]
list_of_detections = CMT_multi_tracker.grab_tracks(format='tlwh') 

#to get list of detections in [[tl_x, tl_y, br_x, br_y], [tl_x, tl_y, br_x, br_y], ...] 
list_of_detections2 = CMT_multi_tracker.grab_tracks(format='tlbr') 
```

To draw the detections on the image:

```python
#draw_image is a opencv BGR image
CMT_multi_tracker.draw_tracks(draw_image)
```

To clear all tracks on the CMT MOT:

```python
CMT_multi_tracker.erase_tracks()
```
