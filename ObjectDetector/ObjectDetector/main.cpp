#pragma
#include "training.h"
#include "svm_detector.h"

void exec_train_svm(const string & pos_dir, const string & pos, const string & neg_dir, const string & neg);
void exec_detect_pic(const string & svm_path);

int main(int argc, char** argv)
{
	const string keys = 
		"{help h | | show help message }"
		"{t train| | train svm         }"
		"{d detect| | detect pic use svm }"
		"{pd     | | pos_dir           }"
		"{p      | | pos.lst           }"
		"{nd     | | neg_dir           }"
		"{n      | | neg.lst           }"
		"{f file | | file to open      }"
		;
	CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		parser.printMessage();
		exit(0);
	}
	// train svm
	if (parser.has("train"))
	{
		string pos_dir = parser.get<string>("pd");
		string pos = parser.get<string>("p");
		string neg_dir = parser.get<string>("nd");
		string neg = parser.get<string>("n");
#ifdef _DEBUG
		pos_dir = "g:\\opencvtest\\poss\\";
		pos = "pos.txt";		
		neg_dir = "g:\\opencvtest\\neg\\";		
		neg = "neg.txt";
#endif
		if (pos_dir.empty() || pos.empty() || neg_dir.empty() || neg.empty())
		{
			cout << "Wrong number of parameters." << endl
				<< "Usage: " << "ObjectDetector" << "-t --pd=pos_dir -p=pos.lst --nd=neg_dir -n=neg.lst" << endl
				<< "example: " << "ObjectDetector" << "-t --pd=/INRIA_dataset/ -p=Train/pos.lst --nd=/INRIA_dataset/ -n=Train/neg.lst" << endl;
			exit(-1);
		}

		exec_train_svm(pos_dir, pos, neg_dir, neg);
	}
	//detect object in picture
	else if (parser.has("detect"))
	{
		string svm_path = parser.get<string>("file");
#ifdef _DEBUG
		svm_path = "g:\\test.yml";
#endif
		if (svm_path.empty())
		{
			cout << "Wrong number of parameters" << endl;
			cout << "Usage: ObjectDetector -d -f=svm.yml" << endl;
		}

		exec_detect_pic(svm_path);
	}
	
	return 0;
}

void exec_train_svm(const string & pos_dir, const string & pos, const string & neg_dir, const string & neg)
{
	cout << "Start training svm, pls wait..." << endl;
	Ptr<SVM> pSvm = train_svm_from(pos_dir, pos, neg_dir, neg);
	cout << "DONE!" << endl;

	string file_path;
	cout << "Input a file path to save the trained svm(end with .yml, ex: my_detector.yml): " << endl;
	cin >> file_path;
	save_svm_to_file(pSvm, file_path);
	cout << "Save svm to " << file_path << " successfully." << endl;
}


void exec_detect_pic(const string & svm_path)
{

	HOGDescriptor detector;
	Size default_size(96, 160);
	build_hog_detector_from_svm(detector, svm_path, default_size);

	
	vector< Rect > locations;
	for (;;)
	{
		cout << "Input the tmage path needed to detect(input \"quit\" to quit): ";
		string img_path;
		cin >> img_path;

		if (img_path.empty() || img_path == "quit")
		{
			clog << "Quit" << endl;
			break;
		}

		locations.clear();
		Mat img = imread(img_path);
		if (img.empty())
		{
			cerr << "Can not open image: " << img_path << endl;
			break;
		}
		//detect target objs in the img
		detect(detector, img, locations);

		//draw the locations with Green color Rect
		draw_locations(img, locations, Scalar(0, 255, 0));

		namedWindow("detect_result", WINDOW_NORMAL);
		imshow("detect_result", img);

		waitKey(0);
	}
}