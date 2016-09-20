#include "svm_detector.h"

/*
* draw rectangles which included in locations on img with color  
*/
void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color)
{
	if (!locations.empty() && !img.empty()) 
	{
		vector< Rect >::const_iterator it = locations.begin();
		vector< Rect >::const_iterator end = locations.end();
		for (; it != end; ++it) 
		{
			//draw rect
			rectangle(img, *it, color, 2);
		}
	}
}

/*
* get hog support vector from trained svm
*/
void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector)
{
	// get the support vectors
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}


/*
* load SVM from yml file
*/
bool load_SVM_from_file(Ptr<SVM> & pSVM, const string & file_name)
{
	pSVM = StatModel::load<SVM>(file_name);
	return pSVM->empty() ? false : true;
}


/*
* config hog detector with trained SVM and window size
*/
void config_detector(HOGDescriptor & detector, Ptr<SVM> & pSvm, const Size & size)
{
	vector< float > hog_detector;
	if (pSvm.empty())
	{
		cerr << "SVM should not be empty." << endl;
		exit(-1);
	}
	
	detector.winSize = size;
	// get detector from trained SVM 
	get_svm_detector(pSvm, hog_detector);
	// config detector with SVM hog detector
	detector.setSVMDetector(hog_detector);

	detector.winSize = size;
}


void detect(HOGDescriptor & detector, const Mat & img, vector< Rect > & locations)
{
	detector.detectMultiScale(img, locations);
}

void build_hog_detector_from_svm(HOGDescriptor & detector, const string & svm_file_path, const Size & size)
{
	Ptr<SVM> pSvm;
	if (!load_SVM_from_file(pSvm, svm_file_path))
	{
		cerr << "Can not open SVM yml file: " << svm_file_path << endl;
		exit(-1);
	}

	config_detector(detector, pSvm, size);
}