#include "hog.h"
#include <cmath>
#include <iostream>
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include <math.h>

//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/video/video.hpp>
//#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

//Martina
//constants for gradient computation
int ddepth = CV_16S;
int scale = 1;
int delta = 0;
Mat kernel = {1, 0, -1};

//constants for histogramm calculation
int hist_bins = 9;
int hist_bin_size = 20;
float hist_ranges[] = {0, 180};
int cell_size = 8;

class HOG::HOGPimpl {
public:

	cv::Mat1f descriptors;
	cv::Mat1f responses;
	
	cv::SVM svm;
	cv::HOGDescriptor hog;

    
};


/// Constructor
HOG::HOG()
{
	pimpl = std::shared_ptr<HOGPimpl>(new HOGPimpl());
}

/// Destructor
HOG::~HOG() 
{
}

/// Start the training.  This resets/initializes the model.
void HOG::startTraining()
{
}

/// Add a new training image.
///
/// @param img:  input image
/// @param bool: value which specifies if img represents a person
void HOG::train(const cv::Mat3b& img, bool isPerson)
{
	cv::Mat3b img2 = img(cv::Rect((img.cols-64)/2,(img.rows-128)/2,64,128));
	vector<float> vDescriptor;
	pimpl->hog.compute(img2, vDescriptor);	
	cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);
    
	pimpl->descriptors.push_back(descriptor);
	pimpl->responses.push_back(cv::Mat1f(1,1,float(isPerson)));
    
    
    // Martina HOC
    
    //Gaussian Blur?! -> worse performance according to paper
    
    //GaussianBlur( img2, img2, Size(3,3), 0, 0, BORDER_DEFAULT );

    //Compute Gradient
    
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat grad, imgGray;

    //Sobel -> worse performance apparently.
    /*
    /// Gradient X
    Sobel( img2, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    /// Gradient Y
    Sobel( img2, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    //convert back to CV_8U
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );
    //add both gradient direction (only approximation of gradient)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    //imshow( "Grad", grad );*/

    //gradient
    cv::cvtColor(img2, imgGray, CV_BGR2GRAY);
    cv::filter2D(img2, grad_x, ddepth, kernel, {-1, -1}, delta, BORDER_DEFAULT);
    cv::filter2D(img2, grad_y, ddepth, kernel.t(), {-1, -1}, delta, BORDER_DEFAULT);

    //Magnitude and angle => look up appropriate functions for opencv on internet
    Mat magnit, angle;
    magnitude(grad_x, grad_y, magnit);

    divide(grad_x, grad_y, angle);
    
    //compute histogramms for cells of 8x8
    //float hists[][][];
    
    //go through all pixels and put them into the neighbouring bins (weighted according to distance from cell center)
    for (int row = 0; row < magnit.rows; ++row) {
        for (int col = 0; col < magnit.cols; ++col) {
            //take atan of each pixel (atan(-180 bis 180) oder atan2 (0 bis 360)?)
            angle.at<float>(row, col) = abs(atan(angle.at<float>(row, col)));
            
            float ang = angle.at<float>(row, col);
            // linear interpolation: weights for angular bins
            float middle_bin1 = (ang/hist_bin_size - 1) * 20 + 10;
            float middle_bin2 = (ang/hist_bin_size) * 20 + 10;
            float weight_bin1 = 1 - (middle_bin1 - ang)/hist_bin_size;
            float weight_bin2 = 1 - (middle_bin2 - ang)/hist_bin_size;
            //bilinear interpolation: weights for cells
            
        }
    }

    
    
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
void HOG::finishTraining()
{
	cv::SVMParams params;
	pimpl->svm.train( pimpl->descriptors, pimpl->responses, cv::Mat(), cv::Mat(), params );
}

/// Classify an unknown test image.  The result is a floating point
/// value directly proportional to the probability of being a person.
///
/// @param img: unknown test image
/// @return:    probability of human likelihood
double HOG::classify(const cv::Mat3b& img)
{
	

	cv::Mat3b img2 = img(cv::Rect((img.cols-64)/2,(img.rows-128)/2,64,128));

	vector<float> vDescriptor;
	pimpl->hog.compute(img2, vDescriptor);	
	cv::Mat1f descriptor(1,vDescriptor.size(),&vDescriptor[0]);

	return -pimpl->svm.predict(descriptor, true);
}

