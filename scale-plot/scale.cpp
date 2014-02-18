#include <opencv2/highgui.hpp>

static cv::Scalar randomColor(void)
{
    static cv::RNG rng;
    const uchar red   = uchar(rng);
    const uchar green = uchar(rng);
    const uchar blue  = uchar(rng);
    return cv::Scalar(blue, green, red);
}

// Draw on image a point at p in color c.
//
static void drawPoint(cv::Mat &image, const cv::Point2f &p, const cv::Scalar &c)
{
    static const int radius = 2;
    static const int thickness = cv::FILLED;
    static const int lineKind = cv::LINE_8;
    static const int shift = 0;
    cv::circle(image, p, radius, c, thickness, lineKind, shift);
}


// Draw rectangle r in color c on image i.
//
static void drawRectangle(cv::Mat &i, const cv::Scalar &c, const cv::Rect &r)
{
    static const int thickness = 1;
    static const int lineKind = cv::LINE_8;
    static const int shift = 0;
    cv::rectangle(i, r, c, thickness, lineKind, shift);
}


int main(int, const char *[])
{
    static const cv::Scalar white(255, 255, 255);
    static const float scales[] = {
        0.16151, 0.19381, 0.23257, 0.27908, 0.33490,
        0.40188, 0.48225, 0.57870, 0.69444, 0.83333,
        1.00000,
        1.20000, 1.44000, 1.72800, 2.07360, 2.48832,
        2.98598, 3.58318, 4.29982, 5.15978, 6.19174
    };
    static const size_t count = sizeof scales / sizeof scales[0];
    static const cv::Size box(50, 50);
    cv::Mat_<cv::Vec3b> image = cv::Mat::zeros(700, 700, CV_8UC3);
    const int x = (image.size().width  - box.width)  / 2;
    const int y = (image.size().height - box.height) / 2;
    const cv::Point origin(x, y);
    const cv::Rect roi(origin, box);
    for (int i = 0; i < count; ++i) {
        const float scale = scales[i];
        const cv::Point q(roi.size());  // Scale the roi size via a Point q.
        const cv::Point p(q * scale);
        const cv::Size sz(p);
        const cv::Rect r(p, sz);
        const cv::Scalar color = scale == 1.0 ? white : randomColor();
        drawRectangle(image, color, r);
    }
    drawRectangle(image, white, roi);
    cv::imshow("Grid Scale Plot", image);
    cv::imwrite("result.png", image);
    cv::waitKey(0);
}
