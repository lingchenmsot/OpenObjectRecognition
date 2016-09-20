#pragma once
#include "training.h"

/*
svm_detector.h contains functions to load the trained SVM, config HOG detector and detect target in images
*/

void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color);
void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
bool load_SVM_from_file(Ptr<SVM> & pSVM, const string & file_name);
void config_detector(HOGDescriptor & detector, Ptr<SVM> & pSvm, const Size & size);
void detect(HOGDescriptor & detector, const Mat & img, vector< Rect > & locations);
void build_hog_detector_from_svm(HOGDescriptor & detector, const string & svm_file_path, const Size & size);



