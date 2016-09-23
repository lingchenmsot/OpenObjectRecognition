#pragma once
#include "training.h"
#include <algorithm>

/*
svm_detector.h contains functions to load the trained SVM, config HOG detector and detect target in images
*/

void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color);
void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
bool load_SVM_from_file(Ptr<SVM> & pSVM, const string & file_name);
void detect(HOGDescriptor & detector, const Mat & img, vector< Rect > & locations);
void config_detector(HOGDescriptor & detector, Ptr<SVM> & pSvm, const Size & win_size, 
	const Size & block_size, const Size & cell_size, const Size & block_stride, int bins);
void build_hog_detector_from_svm(HOGDescriptor & detector, const string & svm_file_path, 
	const Size & win_size, const Size & block_size = Size(16, 16), const Size & cell_size = Size(8, 8),
	const Size & block_stride = Size(8, 8), int bins = 9);

//overlap threshold normally fall in the range 0.3-0.5
void non_max_suppression(const vector< Rect > & srcRects, vector< Rect > & resRects, float overlap_threshold = 0.35);


