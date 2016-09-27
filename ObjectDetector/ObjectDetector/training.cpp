#include "training.h"

/*
* Load images to Mat vector
* prefix: the prefix of the directory.
* filename: the text file containing the fileName of each image.
*/
void load_images(const string & prefix, const string & filename, vector< Mat > & img_lst)
{
	string line;
	ifstream file;

	file.open(prefix + filename);

	if (!file.is_open())
	{
		cerr << "Unable to open the list of images from " << filename << " filename." << endl;
		exit(-1);
	}

	bool end_of_parsing = false;
	while (!end_of_parsing)
	{
		getline(file, line);
		// no more file to read
		if (line.empty()) 
		{
			end_of_parsing = true;
			break;
		}
		// load the image
		Mat img = imread((prefix + line).c_str()); 
		// invalid image, just skip it.
		if (img.empty()) 
			continue;
		
#ifdef _DEBUG
		imshow("image", img);
		waitKey(10);
#endif
		img_lst.push_back(img.clone());
	}
}

/*
* Compute hog of each image
* img_lst: the input image list
* grdient_lst: the hog freature list
* size: hog method window size
*/
void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size)
{
	HOGDescriptor hog;
	hog.winSize = size;
	Mat gray;
	vector< Point > location;
	vector< float > descriptors;

	vector< Mat >::const_iterator img = img_lst.begin();
	vector< Mat >::const_iterator end = img_lst.end();
	for (; img != end; ++img)
	{
		cvtColor(*img, gray, COLOR_BGR2GRAY);
		hog.compute(gray, descriptors, Size(8, 8), Size(0, 0), location);
		gradient_lst.push_back(Mat(descriptors).clone());
		
#ifdef _DEBUG
		imshow("gradient", get_hogdescriptor_visu(img->clone(), descriptors, size));
		waitKey(10);
#endif
		
	}
}

/*
* Sample rect small images from full_neg_lst
* full_neg_lst: full negative image list
* neg_list: sampled rect image list
* size: the sample size
*/
void sample_neg(const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size)
{
	Rect box;
	box.width = size.width;
	box.height = size.height;

	const int size_x = box.width;
	const int size_y = box.height;

	vector< Mat >::const_iterator img = full_neg_lst.begin();
	vector< Mat >::const_iterator end = full_neg_lst.end();
	
	for (; img != end; ++img)
	{
		assert(img->cols >= size_x && img->rows >= size_y);
		for (int cur_x = 0; cur_x < img->cols - size_x; cur_x += size_x)
		{
			for (int cur_y = 0; cur_y < img->rows - size_y; cur_y += size_y)
			{
				box.x = cur_x;
				box.y = cur_y;
				Mat roi = (*img)(box);
				neg_lst.push_back(roi.clone());
#ifdef _DEBUG
				imshow("img", roi.clone());
				waitKey(10);
#endif
			}
		}
	}
}

/*
* Randomly sample rect small images from full_neg_lst
* full_neg_lst: full negative image list
* neg_list: sampled rect image list
* size: the sample size
*/
void random_sample_neg(const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size)
{
	Rect box;
	box.width = size.width;
	box.height = size.height;

	const int size_x = box.width;
	const int size_y = box.height;

	vector< Mat >::const_iterator img = full_neg_lst.begin();
	vector< Mat >::const_iterator end = full_neg_lst.end();

	for (; img != end; ++img)
	{
		assert(img->cols >= size_x && img->rows >= size_y);
		box.x = img->cols == size_x ? 0 : rand() % (img->cols - size_x);
		box.y = img->rows == size_y ? 0 : rand() % (img->rows - size_y);
		Mat roi = (*img)(box);
		neg_lst.push_back(roi.clone());
#ifdef _DEBUG
		imshow("img", roi.clone());
		waitKey(10);
#endif
	}
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData)
{
	const int rows = (int)train_samples.size();
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
	//< used for transposition if needed
	cv::Mat tmp(1, cols, CV_32FC1); 
	trainData = cv::Mat(rows, cols, CV_32FC1);
	vector< Mat >::const_iterator itr = train_samples.begin();
	vector< Mat >::const_iterator end = train_samples.end();
	for (int i = 0; itr != end; ++itr, ++i)
	{
		CV_Assert(itr->cols == 1 ||
			itr->rows == 1);
		if (itr->cols == 1)
		{
			transpose(*(itr), tmp);
			tmp.copyTo(trainData.row(i));
		}
		else if (itr->rows == 1)
		{
			itr->copyTo(trainData.row(i));
		}
	}
}

/*
* Train SVM with training set [gradient_file, labels]
* For positve case, label equals 1, otherwise label equals -1.
*/
Ptr<SVM> train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels)
{
	Mat train_data;
	convert_to_ml(gradient_lst, train_data);

	clog << "Start training...";
	Ptr<SVM> svm = SVM::create();
	// Default values to train SVM
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm->setGamma(0);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(0.01); // From paper, soft classifier
	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task

	svm->train(train_data, ROW_SAMPLE, Mat(labels));
	clog << "...[done]" << endl;
	

	return svm;
}

/*
* Save trained SVM to yml file.
*/
void save_svm_to_file(Ptr<SVM> pSvm, const string & fileName)
{
	if (pSvm.empty()) 
	{
		cerr << "The SVM needed to save is NULL." << endl;
		exit(-1);
	}
	if (fileName.empty()) 
	{
		cerr << "SVM file name should not be empty." << endl;
		exit(-1);
	}

	pSvm->save(fileName);
}

/*
* train a new SVM with provided pos/neg training data
* note: all positive pictures should be the same size.
*/
Ptr<SVM> train_svm_from(const string & pos_dir, const string & pos, const string & neg_dir, const string & neg, const Size & size)
{
	//positve case image list
	vector< Mat > pos_lst;
	//negative case image list
	vector< Mat > full_neg_lst;
	vector< Mat > neg_lst;
	//gradient list of all training images
	vector< Mat > gradient_lst; 
	//pos/neg label, +1 for positive, -1 for negative
	vector< int > labels; 

	// load positive images
	clog << "Loading pos images..." << endl;
	load_images(pos_dir, pos, pos_lst);
	labels.assign(pos_lst.size(), +1);

	//load negative images
	const unsigned int old_size = (unsigned int)labels.size();
	clog << "loading neg images..." << endl;
	load_images(neg_dir, neg, full_neg_lst);
	
	clog << "sampling neg images..." << endl;
	float neg_pos_ratio = full_neg_lst.size() / pos_lst.size();
	random_sample_neg(full_neg_lst, neg_lst, size);
	/*if (neg_pos_ratio > NEG_POS_RATIO_THRESHOLD) {
		random_sample_neg(full_neg_lst, neg_lst, size);
	}
	else
	{
		sample_neg(full_neg_lst, neg_lst, size);
	}*/
	
	labels.insert(labels.end(), neg_lst.size(), -1);
	CV_Assert(old_size < labels.size());

	clog << "computing hog..." << endl;
	//compute hog freature
	compute_hog(pos_lst, gradient_lst, size);
	compute_hog(neg_lst, gradient_lst, size);

	clog << "training SVM..." << endl;
	Ptr<SVM> pSvm = train_svm(gradient_lst, labels);

	return pSvm;
}

/*
* visualize the hog feature in the origin image, only used to debug
*/
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = 8;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

																	   // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	  // compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

								   // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;
}
