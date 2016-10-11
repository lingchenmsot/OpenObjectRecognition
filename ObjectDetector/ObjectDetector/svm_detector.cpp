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
* winSize, scale, winStride(step, size), padding, finalThreshold, useMeanShiftGrouping
*/
void config_detector(HOGDescriptor & detector, Ptr<SVM> & pSvm, const Size & win_size,
	const Size & block_size, const Size & cell_size, const Size & block_stride, int bins)
{
	vector< float > hog_detector;
	if (pSvm.empty())
	{
		cerr << "SVM should not be empty." << endl;
		exit(-1);
	}
	
	detector.winSize = win_size;
	detector.blockSize = block_size;
	detector.cellSize = cell_size;
	detector.blockStride = block_stride;
	detector.nbins = bins;

	// get detector from trained SVM 
	get_svm_detector(pSvm, hog_detector);
	// config detector with SVM hog detector
	detector.setSVMDetector(hog_detector);
}

/*
* Use configed detector to detect targets in img, results will be put in locations.
* detector: configed HOGDescriptor
* img: source image
* locations: detection results
* useNMS: using NMS after detection
*/
void detect(HOGDescriptor & detector, const Mat & img, vector< Rect > & locations, bool useNMS,
	const Size & win_stride, const Size & padding, double scale)
{
	vector < Rect > origin_locations;
	//use hog + svm to detect
	detector.detectMultiScale(img, origin_locations, 0, win_stride, padding, scale);
	if (useNMS)
	{
		//filter overlap detections
		non_max_suppression(origin_locations, locations);
	}
	else
	{
		locations.assign(origin_locations.begin(), origin_locations.end());
	}
}

/*
* build a HOGDescripter detector from svm file with parameters.
*/
void build_hog_detector_from_svm(HOGDescriptor & detector, const string & svm_file_path,
	const Size & win_size, const Size & block_size, const Size & cell_size, const Size & block_stride, int bins)
{
	Ptr<SVM> pSvm;
	if (!load_SVM_from_file(pSvm, svm_file_path))
	{
		cerr << "Can not open SVM yml file: " << svm_file_path << endl;
		exit(-1);
	}

	config_detector(detector, pSvm, win_size, block_size, cell_size, block_stride, bins);
}

/*
* Non max suppression to filter overlap detections.
* See: https://github.com/Nuzhny007/Non-Maximum-Suppression/blob/master/main.cpp
*/
void non_max_suppression(const vector< Rect > & srcRects, vector< Rect > & resRects, float overlap_threshold)
{
	const size_t size = srcRects.size();
	if (size == 0)
		return;

	// sort the rects by the bottom-right y coordinate of the Rect
	multimap<int, size_t> idxs;
	for (size_t i = 0; i < size; ++i)
	{
		idxs.insert(pair<int, size_t>(srcRects[i].br().y, i));
	}

	// keep looping while some indexes still remain in the indexes list
	while (idxs.size() > 0)
	{
		//grab the last index in the sorted rect list and add the rect to pick list
		auto last = --end(idxs);
		const Rect & rect1 = srcRects[last->second];
		resRects.push_back(rect1);
		
		//erase the last
		idxs.erase(last);

		for (auto pos = begin(idxs); pos != end(idxs); )
		{
			const Rect & rect2 = srcRects[pos->second];

			float intArea = (rect1 & rect2).area();
			float unionArea = rect1.area() + rect2.area() - intArea;
			float overlap = intArea / unionArea;

			//if there is sufficent overlap, suppress the current bounding box.
			if (overlap > overlap_threshold)
			{
				pos = idxs.erase(pos);
			}
			else
			{
				++pos;
			}
		}
	}
}
