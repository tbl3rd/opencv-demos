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
    static int across = 2;
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

// Return a normalized histogram across binCount bins for plane.
//
static cv::Mat_<float> normalizedHistogram(const cv::Mat &plane, int binCount)
{
    static const cv::Mat noMask;
    static const int     imageCount     = 1;
    static const int     dimensionCount = 1;
    static const int     binCounts[]    = { binCount };
    static const float   histRange[]    = { 0, binCount };
    static const float  *histRanges[]   = { histRange };
    static const bool    uniform        = true;
    static const bool    accumulate     = false;
    cv::Mat_<float>      histogram, result;
    cv::calcHist(&plane, imageCount, 0, noMask, histogram,
                 dimensionCount, binCounts, histRanges, uniform, accumulate);
    static const double alpha = 0;
    static const int normKind = cv::NORM_MINMAX;
    static const int dtype = -1;
    const double beta = plane.rows;
    cv::normalize(histogram, result, alpha, beta, normKind, dtype, noMask);
    return result;
}

// Draw the normalized histogram in color on image.
//
static void drawHistogram(cv::Mat &image,
                          const cv::Mat_<float> &histogram,
                          const cv::Scalar &color)
{
    const int binWidth = cvRound(1.0 * image.cols / histogram.rows);
    cv::Point p0(0, image.rows - cvRound(histogram(0)));
    for (int i = 1; i < histogram.rows; ++i) {
        static const int thickness = 2;
        static const int lineKind = cv::LINE_8;
        static const int shift = 0;
        const cv::Point p1(i * binWidth, image.rows - cvRound(histogram(i)));
        cv::line(image, p0, p1, color, thickness, lineKind, shift);
        p0 = p1;
    }
}

// Return a new image with a histogram of colors in image after displaying
// each channel of image in a separate window.
//
static cv::Mat computeHistogram(const cv::Mat &image)
{
    static const int max = std::numeric_limits<unsigned char>::max();
    enum { BLUE, GREEN, RED, COLORCOUNT };
    static const struct { cv::Scalar value; const char *name; } color[] = {
        [BLUE]  = { cv::Scalar(max,   0,   0), "blue"  },
        [GREEN] = { cv::Scalar(  0, max,   0), "green" },
        [RED]   = { cv::Scalar(  0,   0, max), "red"   }
    };
    static const int binCount = 1 + max; // a bin for each of [0..max]
    cv::Mat result = cv::Mat_<cv::Vec3b>::zeros(image.rows, image.cols);
    cv::Mat plane[COLORCOUNT];
    cv::split(image, plane);
    for (int c = 0; c < COLORCOUNT; ++c) { // for each color ...
        makeWindow(color[c].name, plane[c]);
        const cv::Mat_<float> hist = normalizedHistogram(plane[c], binCount);
        drawHistogram(result, hist, color[c].value);
    }
    return result;
}

int main(int ac, const char *av[])
{
    if (ac == 2) {
        const cv::Mat image = cv::imread(av[1]);
        if (image.data) {
            std::cout << av[0] << ": Press some key to quit." << std::endl;
            makeWindow("Source Image", image, 3);
            const cv::Mat histogram = computeHistogram(image);
            makeWindow("Color Histogram", histogram);
            cv::waitKey(0);
            return 0;
        }
    }
    std::cerr << av[0] << ": Demonstrate histogram equalization."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file."
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/lena.jpg"
              << std::endl << std::endl;
    return 1;
}
