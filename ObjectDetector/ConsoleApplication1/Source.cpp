#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

static void detectAndDraw(const HOGDescriptor &hog, Mat &img)
{
	vector<Rect> found, found_filtered;
	double t = (double)getTickCount();
	// Run the detector with default parameters. to get a higher hit-rate
	// (and more false alarms, respectively), decrease the hitThreshold and
	// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
	hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
	t = (double)getTickCount() - t;
	cout << "detection time = " << (t*1000. / cv::getTickFrequency()) << " ms" << endl;

	for (size_t i = 0; i < found.size(); i++)
	{
		Rect r = found[i];

		size_t j;
		// Do not add small detections inside a bigger detection.
		for (j = 0; j < found.size(); j++)
		if (j != i && (r & found[j]) == r)
			break;

		if (j == found.size())
			found_filtered.push_back(r);
	}

	for (size_t i = 0; i < found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];

		// The HOG detector returns slightly larger rectangles than the real objects,
		// so we slightly shrink the rectangles to get a nicer output.
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
	}
}
int smain(int argc, char** argv)
{

	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	namedWindow("王梧权最伟大", 1);


	// Read current image
	Mat current_image = imread("pic/u=2497031235,3298287131&fm=21&gp=0.jpg");



	detectAndDraw(hog, current_image);

	imshow("王梧权最伟大", current_image);
	int c = waitKey(0) & 255;



	return 0;
}
