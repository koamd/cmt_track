import cv2
import itertools
from numpy import array, zeros, vstack, hstack, math, nan, argsort, median, \
	argmax, isnan, append
import scipy.cluster
import scipy.spatial
import time
import threading

import numpy as np
import cmt_tracker.util as util
import time

#
#The following code was adapted from https://github.com/toinsson/CMT. As the original code is no longer maintained. 
class CMT(object):

	DETECTOR = 'BRISK'
	DESCRIPTOR = 'BRISK'
	DESC_LENGTH = 512
	MATCHER = 'BruteForce-Hamming'
	THR_OUTLIER = 20
	THR_CONF = 0.75
	THR_RATIO = 0.8

	estimate_scale = True
	estimate_rotation = True

	def __init__(self, create_detector = True, track_id = ""):
		
		if create_detector:
			self.detector = cv2.BRISK_create()
			self.descriptor = self.detector

		self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
		self.tl = (nan, nan)
		self.tr = (nan, nan)
		self.br = (nan, nan)
		self.bl = (nan, nan)
		self.track_id = track_id
		self.bb = array([nan, nan, nan, nan])

	def initialize_with_keypoints(self, im_gray0, tl, br, keypoints_cv, descriptors_cv):
		
		self.tl = (tl[0], tl[1])
		self.tr = (br[0], tl[1])
		self.br = (br[0], br[1])
		self.bl = (tl[0], br[1])
		self.has_result = True

		#select the features within the rectangle
		ind = util.in_rect(keypoints_cv, tl, br)
		selected_keypoints_cv = list(itertools.compress(keypoints_cv, ind))
		self.selected_features  = list(itertools.compress(descriptors_cv, ind))

		self.selected_features = np.array(self.selected_features)

		#convert self.selected_features to numpy array
		selected_keypoints = util.keypoints_cv_to_np(selected_keypoints_cv)
		num_selected_keypoints = len(selected_keypoints_cv)

		if num_selected_keypoints == 0:
			raise Exception('No keypoints found in selection')

		# Remember keypoints that are not in the rectangle as background keypoints
		background_keypoints_cv = list(itertools.compress(keypoints_cv, ~ind))
		background_features = list(itertools.compress(descriptors_cv, ~ind))
		_ = util.keypoints_cv_to_np(background_keypoints_cv)

		# Assign each keypoint a class starting from 1, background is 0
		self.selected_classes = array(range(num_selected_keypoints)) + 1
		background_classes = zeros(len(background_keypoints_cv))

		# Stack background features and selected features into database
		self.features_database = vstack((background_features, self.selected_features))

		# Same for classes
		self.database_classes = hstack((background_classes, self.selected_classes))

		# Get all distances between selected keypoints in squareform
		pdist = scipy.spatial.distance.pdist(selected_keypoints)
		self.squareform = scipy.spatial.distance.squareform(pdist)

		# Get all angles between selected keypoints
		angles = np.empty((num_selected_keypoints, num_selected_keypoints))
		for k1, i1 in zip(selected_keypoints, range(num_selected_keypoints)):
			for k2, i2 in zip(selected_keypoints, range(num_selected_keypoints)):

				# Compute vector from k1 to k2
				v = k2 - k1

				# Compute angle of this vector with respect to x axis
				angle = math.atan2(v[1], v[0])

				# Store angle
				angles[i1, i2] = angle

		self.angles = angles

		# Find the center of selected keypoints
		center = np.mean(selected_keypoints, axis=0)

		# Remember the rectangle coordinates relative to the center
		self.center_to_tl = np.array(tl) - center
		self.center_to_tr = np.array([br[0], tl[1]]) - center
		self.center_to_br = np.array(br) - center
		self.center_to_bl = np.array([tl[0], br[1]]) - center

		# Calculate springs of each keypoint
		self.springs = selected_keypoints - center

		# Set start image for tracking
		self.im_prev = im_gray0

		# Make keypoints 'active' keypoints
		self.active_keypoints = np.copy(selected_keypoints)

		# Attach class information to active keypoints
		self.active_keypoints = hstack((selected_keypoints, self.selected_classes[:, None]))

		# Remember number of initial keypoints
		self.num_initial_keypoints = len(selected_keypoints_cv)


	def initialise(self, im_gray0, tl, br):

		# Get initial keypoints in whole image
		keypoints_cv = self.detector.detect(im_gray0)
		selected_keypoints_cv, descriptors_cv = self.descriptor.compute(im_gray0, keypoints_cv)

		#compute all descriptors
		self.initialize_with_keypoints(im_gray0, tl, br, selected_keypoints_cv, descriptors_cv)

	def estimate(self, keypoints):

		center = array((nan, nan))
		scale_estimate = nan
		med_rot = nan

		# At least 2 keypoints are needed for scale
		if keypoints.size > 1:

			# Extract the keypoint classes
			keypoint_classes = keypoints[:, 2].squeeze().astype(np.int) 

			# Retain singular dimension
			if keypoint_classes.size == 1:
				keypoint_classes = keypoint_classes[None]

			# Sort
			ind_sort = argsort(keypoint_classes)
			keypoints = keypoints[ind_sort]
			keypoint_classes = keypoint_classes[ind_sort]

			# Get all combinations of keypoints
			all_combs = array([val for val in itertools.product(range(keypoints.shape[0]), repeat=2)])	

			# But exclude comparison with itself
			all_combs = all_combs[all_combs[:, 0] != all_combs[:, 1], :]

			# Measure distance between allcombs[0] and allcombs[1]
			ind1 = all_combs[:, 0] 
			ind2 = all_combs[:, 1]

			class_ind1 = keypoint_classes[ind1] - 1
			class_ind2 = keypoint_classes[ind2] - 1

			duplicate_classes = class_ind1 == class_ind2

			if not all(duplicate_classes):
				ind1 = ind1[~duplicate_classes]
				ind2 = ind2[~duplicate_classes]

				class_ind1 = class_ind1[~duplicate_classes]
				class_ind2 = class_ind2[~duplicate_classes]

				pts_allcombs0 = keypoints[ind1, :2]
				pts_allcombs1 = keypoints[ind2, :2]

				# This distance might be 0 for some combinations,
				# as it can happen that there is more than one keypoint at a single location
				dists = util.L2norm(pts_allcombs0 - pts_allcombs1)

				original_dists = self.squareform[class_ind1, class_ind2]

				scalechange = dists / original_dists

				# Compute angles
				angles = np.empty((pts_allcombs0.shape[0]))

				v = pts_allcombs1 - pts_allcombs0
				angles = np.arctan2(v[:, 1], v[:, 0])
				
				original_angles = self.angles[class_ind1, class_ind2]

				angle_diffs = angles - original_angles

				# Fix long way angles
				long_way_angles = np.abs(angle_diffs) > math.pi

				angle_diffs[long_way_angles] = angle_diffs[long_way_angles] - np.sign(angle_diffs[long_way_angles]) * 2 * math.pi

				scale_estimate = median(scalechange)
				if not self.estimate_scale:
					scale_estimate = 1;

				med_rot = median(angle_diffs)
				if not self.estimate_rotation:
					med_rot = 0;

				keypoint_class = keypoints[:, 2].astype(np.int)
				votes = keypoints[:, :2] - scale_estimate * (util.rotate(self.springs[keypoint_class - 1], med_rot))

				# Remember all votes including outliers
				self.votes = votes

				# Compute pairwise distance between votes
				pdist = scipy.spatial.distance.pdist(votes)

				# Compute linkage between pairwise distances
				linkage = scipy.cluster.hierarchy.linkage(pdist)

				# Perform hierarchical distance-based clustering
				T = scipy.cluster.hierarchy.fcluster(linkage, self.THR_OUTLIER, criterion='distance')

				# Count votes for each cluster
				cnt = np.bincount(T)  # Dummy 0 label remains
				
				# Get largest class
				Cmax = argmax(cnt)

				# Identify inliers (=members of largest class)
				inliers = T == Cmax
				# inliers = med_dists < THR_OUTLIER

				# Remember outliers
				self.outliers = keypoints[~inliers, :]

				# Stop tracking outliers
				keypoints = keypoints[inliers, :]

				# Remove outlier votes
				votes = votes[inliers, :]

				# Compute object center
				center = np.mean(votes, axis=0)

		return (center, scale_estimate, med_rot, keypoints)

	def process_frame_with_keypoints(self, im_gray, keypoints_cv, features):

		tracked_keypoints, _ = util.track(self.im_prev, im_gray, self.active_keypoints)
		(center, scale_estimate, rotation_estimate, tracked_keypoints) = self.estimate(tracked_keypoints)

		# Create list of active keypoints
		active_keypoints = zeros((0, 3)) 

		# Get the best two matches for each feature
		matches_all = self.matcher.knnMatch(features, self.features_database, 2)
		# Get all matches for selected features
		if not any(isnan(center)):
			selected_matches_all = self.matcher.knnMatch(features, self.selected_features, len(self.selected_features))

		# For each keypoint and its descriptor
		if len(keypoints_cv) > 0:
			transformed_springs = scale_estimate * util.rotate(self.springs, -rotation_estimate)
			for i in range(len(keypoints_cv)):

				# Retrieve keypoint location
				location = np.array(keypoints_cv[i].pt)

				# First: Match over whole image
				# Compute distances to all descriptors
				matches = matches_all[i]
				distances = np.array([m.distance for m in matches])

				# Convert distances to confidences, do not weight
				combined = 1 - distances / self.DESC_LENGTH

				classes = self.database_classes

				# Get best and second best index
				bestInd = matches[0].trainIdx
				secondBestInd = matches[1].trainIdx

				# Compute distance ratio according to Lowe
				ratio = (1 - combined[0]) / (1 - combined[1])

				# Extract class of best match
				keypoint_class = classes[bestInd]

				# If distance ratio is ok and absolute distance is ok and keypoint class is not background
				if ratio < self.THR_RATIO and combined[0] > self.THR_CONF and keypoint_class != 0:

					# Add keypoint to active keypoints
					new_kpt = append(location, keypoint_class)
					active_keypoints = append(active_keypoints, array([new_kpt]), axis=0)

				# In a second step, try to match difficult keypoints
				# If structural constraints are applicable
				if not any(isnan(center)):

					# Compute distances to initial descriptors
					matches = selected_matches_all[i]				
					distances = np.array([m.distance for m in matches])
					# Re-order the distances based on indexing
					idxs = np.argsort(np.array([m.trainIdx for m in matches]))
					distances = distances[idxs]					

					# Convert distances to confidences
					confidences = 1 - distances / self.DESC_LENGTH

					# Compute the keypoint location relative to the object center
					relative_location = location - center

					# Compute the distances to all springs
					displacements = util.L2norm(transformed_springs - relative_location)

					# For each spring, calculate weight
					weight = displacements < self.THR_OUTLIER  # Could be smooth function

					combined = weight * confidences

					classes = self.selected_classes

					# Sort in descending order
					sorted_conf = argsort(combined)[::-1]  # reverse

					# Get best and second best index
					bestInd = sorted_conf[0]
					secondBestInd = sorted_conf[1]

					# Compute distance ratio according to Lowe
					ratio = (1 - combined[bestInd]) / (1 - combined[secondBestInd])

					# Extract class of best match
					keypoint_class = classes[bestInd]

					# If distance ratio is ok and absolute distance is ok and keypoint class is not background
					if ratio < self.THR_RATIO and combined[bestInd] > self.THR_CONF and keypoint_class != 0:

						# Add keypoint to active keypoints
						new_kpt = append(location, keypoint_class)

						# Check whether same class already exists
						if active_keypoints.size > 0:
							same_class = np.nonzero(active_keypoints[:, 2] == keypoint_class)
							active_keypoints = np.delete(active_keypoints, same_class, axis=0)

						active_keypoints = append(active_keypoints, array([new_kpt]), axis=0)

		# If some keypoints have been tracked
		if tracked_keypoints.size > 0:

			# Extract the keypoint classes
			tracked_classes = tracked_keypoints[:, 2]

			# If there already are some active keypoints
			if active_keypoints.size > 0:

				# Add all tracked keypoints that have not been matched
				associated_classes = active_keypoints[:, 2]
				missing = ~np.in1d(tracked_classes, associated_classes)
				active_keypoints = append(active_keypoints, tracked_keypoints[missing, :], axis=0)

			# Else use all tracked keypoints
			else:
				active_keypoints = tracked_keypoints

		# Update object state estimate
		_ = active_keypoints
		self.center = center
		self.scale_estimate = scale_estimate
		self.rotation_estimate = rotation_estimate
		self.tracked_keypoints = tracked_keypoints
		self.active_keypoints = active_keypoints
		self.im_prev = im_gray
		self.keypoints_cv = keypoints_cv
		_ = time.time()

		self.tl = (nan, nan)
		self.tr = (nan, nan)
		self.br = (nan, nan)
		self.bl = (nan, nan)
		self.bb = array([nan, nan, nan, nan])

		self.has_result = False
		if not any(isnan(self.center)) and self.active_keypoints.shape[0] > self.num_initial_keypoints / 5:
			self.has_result = True

			tl = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_tl[None, :], rotation_estimate).squeeze())
			tr = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_tr[None, :], rotation_estimate).squeeze())
			br = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_br[None, :], rotation_estimate).squeeze())
			bl = util.array_to_int_tuple(center + scale_estimate * util.rotate(self.center_to_bl[None, :], rotation_estimate).squeeze())

			min_x = min((tl[0], tr[0], br[0], bl[0]))
			min_y = min((tl[1], tr[1], br[1], bl[1]))
			max_x = max((tl[0], tr[0], br[0], bl[0]))
			max_y = max((tl[1], tr[1], br[1], bl[1]))

			self.tl = tl
			self.tr = tr
			self.bl = bl
			self.br = br

			self.bb = np.array([min_x, min_y, max_x - min_x, max_y - min_y])

	def process_frame(self, im_gray):

		# Detect keypoints, compute descriptors
		keypoints_cv = self.detector.detect(im_gray) 
		keypoints_cv, features = self.descriptor.compute(im_gray, keypoints_cv)

		self.process_frame_with_keypoints(im_gray, keypoints_cv, features)

		

class MultiCMT(object):
	#multi CMT contains a list of CMT object, which each CMT performs tracking on 1 object
	
	def __init__(self):
		#define a list of CMT tracks
		self.detector = cv2.BRISK_create()
		self.descriptor = self.detector
		self.cmt_list = []

	def initialize_track_multi_track(self, im_gray0, bbox: list):
		# Initialise detector, descriptor, matcher
		# check opencv version

		# Get initial keypoints in whole image
		
		keypoints_cv = self.detector.detect(im_gray0)
		keypoints_cv, features = self.descriptor.compute(im_gray0, keypoints_cv)
		image_process_toc = time.time()

		for index,each_bbox in enumerate(bbox):
			#Create a CMT object. 
			tl = [each_bbox[0], each_bbox[1]]
			br = [each_bbox[2], each_bbox[3]]
			cmt_tracker = CMT(create_detector=False)
			cmt_tracker.initialize_with_keypoints(im_gray0, tl, br, keypoints_cv, features)
			self.cmt_list.append(cmt_tracker)
		
	def erase_tracks(self):
		#remove all cmt tracks
		self.cmt_list.clear()

	def process_tracks_with_features(self, im_gray, keypoints_cv, features):
		
		# for cmt_tracks in self.cmt_list:
		# 	cmt_tracks.process_frame_with_keypoints(im_gray, keypoints_cv, features)
		
		thread_list = list()
		for cmt_tracks in self.cmt_list:
			t1 = threading.Thread(target=cmt_tracks.process_frame_with_keypoints, args=(im_gray, keypoints_cv, features))
			thread_list.append(t1)
			t1.start()

		for all_threads in thread_list:
			all_threads.join()

		#check if track should be deleted.
		ind_to_remove = []
		for index, cmt_tracks in enumerate(self.cmt_list):
			if not cmt_tracks.has_result:
				ind_to_remove.append(index)
		
		#delete tracks from the larger index first
		for index in sorted(ind_to_remove, reverse=True):
			del self.cmt_list[index]

	def process_tracks(self, im_gray): 
		"""Runs the tracker and predicts the new position in current frame

		Args:
			im_gray (_type_): input image
		"""
		
		#compute descriptor in current frame
		#feature_extract_tick = time.time()
		keypoints_cv = self.detector.detect(im_gray) 
		keypoints_cv, features = self.descriptor.compute(im_gray, keypoints_cv)
		#feature_extract_tock = time.time()
		#print('[Info] Feature Extract Time: {0}ms'.format((feature_extract_tock-feature_extract_tick)*1000))
       
		#runs all tracker object
		track_tick = time.time()
		self.process_tracks_with_features(im_gray, keypoints_cv, features)
		track_tock = time.time()
		#print('[Info] Tracker Time: {0}ms'.format((track_tock-track_tick)*1000))

	def append_detections(self, im_gray0, bbox: list, labels=[]):
		
		# Get initial keypoints in whole image
		keypoints_cv = self.detector.detect(im_gray0)
		keypoints_cv, features = self.descriptor.compute(im_gray0, keypoints_cv)

		#for each bbox, detect if there is an overlapping tracker. 
		ind_to_remove = []

		#if labels are present, we check for labels rather than aabb overlap. 
		useTrackId = len(labels) > 0

		for bbox_index, each_bbox in enumerate(bbox):
			tl = [each_bbox[0], each_bbox[1]]
			br = [each_bbox[2], each_bbox[3]]

			for index, cmt_tracks in enumerate(self.cmt_list):
				
				if not useTrackId:
					iou_overlap = self.determine_overlap([cmt_tracks.tl[0], cmt_tracks.tl[1]],
														[cmt_tracks.br[0], cmt_tracks.br[1]], tl, br)
					if iou_overlap > 0.2:
						#check if value exist in array
						if index not in ind_to_remove:
							ind_to_remove.append(index)
				else: #replace existing labels. 
					if cmt_tracks.track_id == labels[bbox_index]:
						if index not in ind_to_remove:
							ind_to_remove.append(index)
			
		#delete tracks from the larger index first
		for index in sorted(ind_to_remove, reverse=True):
			del self.cmt_list[index]

		#run and process all existing tracks
		self.process_tracks_with_features(im_gray0, keypoints_cv, features)

		#add all new tracks
		for index, each_bbox in enumerate(bbox):
			tl = [each_bbox[0], each_bbox[1]]
			br = [each_bbox[2], each_bbox[3]]
			#append new detection to tracks
			if len(labels) == 0:
				cmt_tracker = CMT(create_detector=False)
			else:
				cmt_tracker = CMT(create_detector=False, track_id=labels[index])

			cmt_tracker.initialize_with_keypoints(im_gray0, tl, br, keypoints_cv, features)
			self.cmt_list.append(cmt_tracker)

		
	def draw_tracks(self, im_draw):
		for cmt_tracks in self.cmt_list:
			if cmt_tracks.has_result:
				cv2.rectangle(im_draw, cmt_tracks.tl, cmt_tracks.br, (255, 0, 0), 4)
		
	def grab_tracks(self, format='tlwh')-> list:
		"""_summary_

		Returns:
			list: returns a list of tracks [[id, tl_x, tl_y, w, h]]
		"""
		tracker_tracks = []
		for cmt_tracks in self.cmt_list:

			if cmt_tracks.has_result:
				curr_track_id = '0'
				if cmt_tracks.track_id != "":
					curr_track_id = cmt_tracks.track_id

					if format == 'tlwh':
						width = cmt_tracks.br[0] - cmt_tracks.tl[0]
						height = cmt_tracks.br[1] - cmt_tracks.tl[1]
						tracker_tracks.append([curr_track_id, cmt_tracks.tl[0], cmt_tracks.tl[1], width, height])
					elif format == 'tlbr':
						tracker_tracks.append([curr_track_id, cmt_tracks.tl[0], cmt_tracks.tl[1], cmt_tracks.br[0], cmt_tracks.br[1]])

		return tracker_tracks

	def determine_overlap(self, cmt_tl: list, cmt_br: list, tl: list, br: list):
		"""Determine IOU overlap of bounding boxes

		Args:
			cmt_tl (_type_): list [x,y] top left coordinate of CMT tracker
			cmt_br (_type_): list [x,y] bottom right coordinate of CMT tracker
			tl (_type_): list [x,y] top left coordinate
			br (_type_): list [x,y] bottom right coordinate
		"""
		overlap = 0.0
		xA = max(cmt_tl[0], tl[0])
		yA = max(cmt_tl[1], tl[1])
		xB = min(cmt_br[0], br[0])
		yB = min(cmt_br[1], br[1])
		# compute the area of intersection rectangle
		interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = (cmt_br[0] - cmt_tl[0] + 1) * (cmt_br[1] - cmt_tl[1] + 1)
		boxBArea = (br[0] - tl[0] + 1) * (br[1] - tl[1] + 1)
		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)
		# return the intersection over union value
		return iou