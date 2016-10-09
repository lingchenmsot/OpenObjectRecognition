#pragma once
#include <string>
#include <vector>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <time.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

/*
Training.h contains functions to load, tranform training images and train a HOG SVM detector.
*/

void load_images(const string & prefix, const string & filename, vector< Mat > & img_lst);
void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size);
void sample_neg(const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size);
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData);
Ptr<SVM> train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels);
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);
void save_svm_to_file(Ptr<SVM> pSvm, const string & fileName);
Ptr<SVM> train_svm_from(const string & pos_dir, const string & pos, const string & neg_dir, 
	const string & neg, const Size & size);


const static float NEG_POS_RATIO_THRESHOLD = 6.0;