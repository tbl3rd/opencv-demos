#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


// Create a new unobscured named window for image.
// Reset windows layout with when reset is not 0.
//
// The 23 term works around how MacOSX decorates windows.
//
static void makeWindow(const char *window, const cv::Mat &image, int reset = 0)
{
    static int across = 1;
    static int count, moveX, moveY, maxY = 0;
    if (reset) {
        across = reset;
        count = moveX = moveY = maxY = 0;
    }
    if (count % across == 0) {
        moveY += maxY + 23;
        maxY = moveX = 0;
    }
    ++count;
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(window, moveX, moveY);
    cv::imshow(window, image);
    moveX += image.cols;
    maxY = std::max(maxY, image.rows);
}

// Draw circle (x, y, r) on image with a green center at (x, y) and a red
// perimeter at radius r around the center.
//
static void drawCircle(cv::Mat &image, const cv::Vec3f &circle)
{
    std::cout << "circle == " << circle << std::endl;
    const float centerX = circle[0];
    const float centerY = circle[1];
    const float fRadius = circle[2];
    const cv::Point center(cvRound(centerX), cvRound(centerY));
    static const int centerRadius = 3;
    static const cv::Scalar colorGreen(0, 255, 0);
    static const int centerThickness = cv::FILLED;
    static const int lineKind = cv::LINE_8;
    static const int shift = 0;
    cv::circle(image, center, centerRadius, colorGreen,
               centerThickness, lineKind, shift);
    const int outerRadius = cvRound(fRadius);
    static const cv::Scalar colorRed(0, 0, 255);
    static const int outerThickness = 3;
    cv::circle(image, center, outerRadius, colorRed,
               outerThickness, lineKind, shift);
}

// Discover circles in gray using the Hough transform and return a copy of
// image after drawing the discovered circles on it.
//
static cv::Mat drawHoughCircles(const cv::Mat &gray, const cv::Mat &image)
{
    static const int method = cv::HOUGH_GRADIENT;
    static const double dotPitchRatio = 1.0;
    static const double minDistance = 3.0;
    static const double param1 = 200.0;
    static const double param2 = 44.0;
    static const int minRadius = 0;
    static const int maxRadius = 0;
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, method, dotPitchRatio, minDistance,
                     param1, param2, minRadius, maxRadius);
    std::cout << "circles.size() == " << circles.size() << std::endl;
    cv::Mat result;
    image.copyTo(result);
    for (int i = 0; i < circles.size(); ++i) drawCircle(result, circles[i]);
    return result;
}

// Return a grayscale copy of image blurred with a 7x7 kernel.
//
static cv::Mat blurGray(const cv::Mat &image)
{
    cv::Mat gray, result;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    static const cv::Size kernelSize(7, 7);
    static const double sigmaX = 2.0;
    static const double sigmaY = 2.0;
    cv::GaussianBlur(gray, result, kernelSize, sigmaX, sigmaY);
    return result;
}


int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat image = cv::imread(av[1]);
        if (image.data) {
            makeWindow("Original", image, 2);
            const cv::Mat blurred = blurGray(image);
            makeWindow("Blurred Grayscale", blurred);
            const cv::Mat circles = drawHoughCircles(blurred, image);
            makeWindow("Hough Circles", circles);
            cv::waitKey(0);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate circle finding with Hough transform."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/prototype.jpg"
              << std::endl << std::endl;
    return 1;
}
