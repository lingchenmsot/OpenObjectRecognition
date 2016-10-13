#include "main.h"


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
		"{w width| 96 | width of pos sample}"
		"{l length |160 | length of pos sample}"
		;
	
	CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		parser.printMessage();
		exit(0);
	}
	
	int width = parser.get<int>("w");
	int length = parser.get<int>("l");
	Size size(width, length);
	// train svm
	if (parser.has("train"))
	{
		string pos_dir = parser.get<string>("pd");
		string pos = parser.get<string>("p");
		string neg_dir = parser.get<string>("nd");
		string neg = parser.get<string>("n");
		
		/*
		pos_dir = "g:\\opencvtest\\poss\\";
		pos = "pos.txt";
		neg_dir = "e:\\images\\neg\\";
		//neg_dir = "E:\\Image\\RedKele96x160\\";
		neg = "neg.txt";
		*/

		if (pos_dir.empty() || pos.empty() || neg_dir.empty() || neg.empty())
		{
			cout << "Wrong number of parameters." << endl
				<< "Usage: " << "ObjectDetector" << "-t --pd=pos_dir -p=pos.lst --nd=neg_dir -n=neg.lst -w=width -l=length" << endl
				<< "example: " << "ObjectDetector" 
				<< "-t --pd=/INRIA_dataset/ -p=Train/pos.lst --nd=/INRIA_dataset/ -n=Train/neg.lst -w=96 -l=160" << endl;
			exit(-1);
		}

		exec_train_svm(pos_dir, pos, neg_dir, neg, size);
	}
	//detect object in picture
	else if (parser.has("detect"))
	{
		string svm_path = parser.get<string>("file");
		//svm_path = "g:\\l_m.yml";
		if (svm_path.empty())
		{
			cout << "Wrong number of parameters" << endl;
			cout << "Usage: ObjectDetector -d -f=svm.yml -w=96 -l=160" << endl;
			exit(-1);
		}

		exec_detect_pic(svm_path, size);
	}
	
	return 0;
}

void exec_train_svm(const string & pos_dir, const string & pos, const string & neg_dir, const string & neg, const Size & cell_size)
{
	cout << "Start training svm, pls wait..." << endl;
	Ptr<SVM> pSvm = train_svm_from(pos_dir, pos, neg_dir, neg, cell_size);
	cout << "DONE!" << endl;

	string file_path;
	cout << "Input a file path to save the trained svm(end with .yml, ex: my_detector.yml): " << endl;
	cin >> file_path;
	save_svm_to_file(pSvm, file_path);
	cout << "Save svm to " << file_path << " successfully." << endl;
}


void exec_detect_pic(const string & svm_path, const Size & cell_size)
{

	HOGDescriptor detector;
	Size default_size = cell_size;
	clog << "loading SVM: " << svm_path << ", and building detector..." << endl;
	build_hog_detector_from_svm(detector, svm_path, default_size);
	
	vector< Rect > locations;
	for (;;)
	{
		cout << "Input the image path needed to detect(input \"quit\" to quit): ";
		string img_path;
		cin >> img_path;

		if (img_path.empty() || img_path == "quit")
		{
			clog << "Quit" << endl;
			exit(-1);
		}

		locations.clear();
		Mat img = imread(img_path);
		if (img.empty())
		{
			cerr << "Can not open image: " << img_path << endl;
			break;
		}

		cout << "detecting..." << endl;
		//detect target objs in the img
		detect(detector, img, locations, true, Size(8, 16), Size(30, 30), 1.10);

		//draw the locations with Green color Rect
		draw_locations(img, locations, Scalar(0, 255, 0));
		cout << "complete!" << endl;

		namedWindow("detect_result", WINDOW_NORMAL);
		imshow("detect_result", img);

		waitKey(0);
	}
}

void get_hard_examples(const string & output_dir, const string & svm_path, const string & dir,
	const string & names, Size& size)
{
	HOGDescriptor detector;
	clog << "loading SVM: " << svm_path << ", and building detector..." << endl;
	build_hog_detector_from_svm(detector, svm_path, size);

	clog << "starting to detecting..." << endl;
	int count = 0;
	vector< Rect > locations;

	string line;
	ifstream file(dir + names);
	if (!file.is_open())
	{
		cerr << "Unable to open the list of images from " << dir << names << endl;
		exit(-1);
	}

	bool end = false;
	while (!end)
	{
		getline(file, line);
		if (line.empty()) {
			end = true;
			break;
		}
		Mat img = imread(dir + line);
		if (img.empty()) continue;

		locations.clear();
		detect(detector, img, locations, false);
		for (vector< Rect >::iterator rect = locations.begin(); rect != locations.end(); ++rect)
		{
			clog << "false detection in " << dir + line << endl;
			//keep the rect stay in the origin image
			rect->x = rect->x < 0 ? 0 : rect->x;
			rect->y = rect->y < 0 ? 0 : rect->y;
			if (rect->x + rect->width > img.cols)
				rect->width = img.cols - rect->x;
			if (rect->y + rect->height > img.rows)
				rect->height = img.rows - rect->y;
			//crop false detected img from the origin
			Mat hard_example = img(*rect);
			//resize img to default size
			resize(hard_example, hard_example, size);

			//write the img to file
			++count;
			stringstream ss;
			ss << output_dir + line.substr(0, line.find_last_of('.')) + "_hard_" <<  count << ".png";
			string file_name;
			ss >> file_name;
			clog << "writing hard example image to : " << file_name << endl;
			imwrite(file_name, hard_example);
		}
	}
}