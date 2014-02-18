﻿#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sstream>

// Show a usage message on cout for program named av0.
//
static void showUsage(const char *av0)
{
    std::cout
        << av0 << ": Time scanning a Mat with the C operator[] method, "
        << std::endl
        << "    matrix iterators, the at() function, and the LUT() function."
        << std::endl << std::endl
        << "Usage: " << av0 << " <image-file> <divisor> [g]"
        << std::endl << std::endl
        << "Where: <image-file> is the path to an image file."
        << std::endl
        << "       The image should have a Mat::depth() of CV_8U."
        << std::endl
        << "       <divisor> is a small integer less than 255."
        << std::endl
        << "       g means process the image in gray scale."
        << std::endl << std::endl
        << "Example: " << av0 << " ../resources/Twas_Ever_Thus500.jpg 10"
        << std::endl
        << "Read an image object from Twas_Ever_Thus500 into a cv::Mat."
        << std::endl
        << "Repeatedly divide the image's native color palette by 10."
        << std::endl << std::endl;
}

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

// Return divisor after loading an image file into img.
// Return 0 after showing a usage message if there's a problem.
//
static int useCommandLine(int ac, const char *av[], cv::Mat &img)
{
    if (ac > 2) {
        int divisor = 0;
        std::stringstream ss; ss << av[2]; ss >> divisor;
        if (ss && divisor) {
            const bool g = (ac == 4) && (*"g" == *av[3]);
            const int cogOpt = g? cv::IMREAD_GRAYSCALE: cv::IMREAD_COLOR;
            img = cv::imread(av[1], cogOpt);
            if (img.data) return divisor;
        }
    }
    showUsage(av[0]);
    return 0;
}


// Call scan(*this) a lot and report the average run time in milliseconds.
//
// After cloning image, scan() visits every element, reduces it according
// to table, and returns the result.
//
// Each test() runs scan() many times and reports its average run time.
//
struct Test {
    const cv::Mat &table;
    const cv::Mat &image;
    const char *const label;
    cv::Mat (*scan)(const struct Test &);

    void operator()(void) const {
        static const int runCount = 200;
        cv::Mat reduced;
        const int64 tickZero = cv::getTickCount();
        for (int i = 0; i < runCount; ++i) reduced = (*scan)(*this);
        const int64 ticks = cv::getTickCount() - tickZero;
        const double totalSeconds = (double)ticks / cv::getTickFrequency();
        const double msPerRun = totalSeconds * 1000 / runCount;
        std::cout << "Average " << label << " time in milliseconds: "
                  << msPerRun << std::endl;
        makeWindow(label, reduced);
    }

    Test(const cv::Mat &lut, const cv::Mat &i, const char *m,
         cv::Mat (*s)(const struct Test &)):
        table(lut), image(i), label(m), scan(s) {}
};


// Scan t.image using C's native array [] on rows pulled via Mat::ptr<>(),
// while also pulling a native lookup table from t.table.data.  This is
// generally the most efficient scanning method, and can be even better
// if the image.isContinuous().
//
// With care, this is as effective as a random access iterator, but using
// LUT() as in scanWithLut() is much more convenient and about as fast when
// processing an entire matrix with a lookup table.
//
static cv::Mat scanWithArrayOp(const Test &t)
{
    cv::Mat image = t.image.clone();
    int nRows = image.rows;
    int nCols = image.cols * image.channels();
    if (image.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }
    const uchar *const table = t.table.data;
    for (int i = 0; i < nRows; ++i) {
        uchar *const p = image.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j) p[j] = table[p[j]];
    }
    return image;
}


// Scan t.image using MatIterator<>, pulling a native lookup table from
// t.table.data.  The iterator knows the dimensions of the image matrix,
// but not its channels().  This is slower but safer than
// scanWithArrayOp(), and has performance similar to scanWithAt().
//
static cv::Mat scanWithMatIter(const Test &t)
{
    cv::Mat image = t.image.clone();
    const uchar *const table = t.table.data;
    switch (image.channels()) {
    case 1: {
        cv::MatIterator_<uchar> it = image.begin<uchar>();
        const cv::MatIterator_<uchar> end = image.end<uchar>();
        for ( ; it != end; ++it) *it = table[*it];
        break;
    }
    case 3: {
        cv::MatIterator_<cv::Vec3b> it = image.begin<cv::Vec3b>();
        const cv::MatIterator_<cv::Vec3b> end = image.end<cv::Vec3b>();
        for( ; it != end; ++it) {
            (*it)[0] = table[(*it)[0]];
            (*it)[1] = table[(*it)[1]];
            (*it)[2] = table[(*it)[2]];
        }
    }
    }
    return image;
}


// Treat the matrix like a multi-dimensional array.
//
// Scan t.image using Mat::at<>() or Mat_::operator()(), pulling a native
// lookup table from t.table.data.  As with the iterator scan, these know
// the dimensions of the image matrix but not its channels().  You can
// specify the element type on each access using the Mat::at<>() member
// template, or you can specify it once by embedding the matrix data in a
// new header with the Mat_<>() class template which takes an element type.
//
// So scanWithAt() uses Mat::at<>() for grayscale and Mat_<>() for color.
//
// This performs comparable to scanWithMatIter() and is more convenient
// for random access modification of an image rather than scanning it.
//
static cv::Mat scanWithAt(const Test &t)
{
    cv::Mat image = t.image.clone();
    const uchar *const table = t.table.data;
    switch (image.channels()) {
    case 1: {
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                image.at<uchar>(i,j) = table[image.at<uchar>(i,j)];
            }
        }
        break;
    }
    case 3: {
        cv::Mat_<cv::Vec3b> head = image;
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                head(i,j)[0] = table[head(i,j)[0]];
                head(i,j)[1] = table[head(i,j)[1]];
                head(i,j)[2] = table[head(i,j)[2]];
            }
        }
        image = head;
        break;
    }
    }
    return image;
}


// Scan t.image using LUT() and t.table directly.
//
// LUT() is almost as efficient as scanWithArrayOp() when scanning an
// entire matrix, and is most convenient when there is a lookup table
// already computed.
//
static cv::Mat scanWithLut(const Test &t)
{
    cv::Mat image = t.image.clone();
    LUT(t.image, t.table, image);
    return image;
}


int main(int ac, const char *av[])
{
    cv::Mat image, table(1, 256, CV_8U);
    const int divisor = useCommandLine(ac, av, image);
    if (divisor == 0 || CV_8U != image.depth()) return 1;
    makeWindow(av[1], image, 3);
    uchar *const p = table.data;
    for (int i = 0; i < table.cols; ++i) p[i] = (divisor * (i / divisor));
    const Test tests[] = {
        Test(table, image, "operator[]", &scanWithArrayOp),
        Test(table, image, "iterator  ", &scanWithMatIter),
        Test(table, image, "at()      ", &scanWithAt),
        Test(table, image, "LUT()     ", &scanWithLut)
    };
    const int testsCount = sizeof tests / sizeof tests[0];
    for (int i = 0; i < testsCount; ++i) (tests[i])();
    cv::waitKey(0);
    return 0;
}
