#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


static void showUsage(const char *av0)
{
    std::cerr << std::endl
              << av0 << ": Demonstrate Delaunay triangulation "
              << "and Voronoi tesselation." << std::endl << std::endl
              << "Usage: " << av0 << std::endl << std::endl;
}

static void showKeys(const char *av0)
{
    std::cerr << av0 << ": Updating the mesh with successive random points."
              << std::endl
              << av0 << ": Press any key to stop adding points." << std::endl
              << av0 << ": Then press any key again to quit." << std::endl;
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

// Return a random BGR color.
//
static cv::Scalar randomColor(void)
{
    static cv::RNG rng;
    const uchar blue  = uchar(rng);
    const uchar green = uchar(rng);
    const uchar red   = uchar(rng);
    return cv::Scalar(blue, green, red);
}

// Return a random point in r excluding some border margin.
//
static cv::Point2f randomPoint(const cv::Size &s)
{
    static cv::RNG rng;
    static const float margin = 10.0;
    const float x = rng.uniform(margin, 1.f * s.width  - margin);
    const float y = rng.uniform(margin, 1.f * s.height - margin);
    const cv::Point2f result(x, y);
    return result;
}

// Draw on image a point at p in color c.
//
static void drawPoint(cv::Mat &image, const cv::Point2f &p, const cv::Scalar &c)
{
    static const int radius = 3;
    static const int thickness = cv::FILLED;
    static const int lineKind = cv::LINE_8;
    static const int shift = 0;
    cv::circle(image, p, radius, c, thickness, lineKind, shift);
}

// Draw on image, a line from p0 to p1 in color c.
//
static void drawLine(cv::Mat &image, const cv::Point &p0, const cv::Point &p1,
                     const cv::Scalar &c)
{
    static const int thickness = 1;
    static const int lineKind = cv::LINE_AA;
    static const int shift = 0;
    cv::line(image, p0, p1, c, thickness, lineKind, shift);
}

// Draw sd edges on image in white.
//
static void drawDelaunay(cv::Mat &image, const cv::Subdiv2D &sd)
{
    static const cv::Scalar white(255, 255, 255);
    std::vector<cv::Vec4f> edges;
    sd.getEdgeList(edges);
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const cv::Vec4f &e = edges[i];
        const cv::Point p0(cvRound(e[0]), cvRound(e[1]));
        const cv::Point p1(cvRound(e[2]), cvRound(e[3]));
        drawLine(image, p0, p1, white);
    }
}

// Locate new random point p in sd and outline its triangle (if any) on
// image in red.  Return the new point's index in sd.
//
// Find some index (edge0) of the triangle and draw it.
// Then draw edges around left until arriving back at edge0.
//
static int addRandomPoint(cv::Mat &image, cv::Subdiv2D &sd)
{
    static const cv::Scalar red(0, 0, 255);
    const cv::Point2f p = randomPoint(image.size());
    drawPoint(image, p, red);
    int edge0, vertex0ignored;
    sd.locate(p, edge0, vertex0ignored);
    if (edge0 > 0) {
        int e = edge0;
        while (true) {
            cv::Point2f org, dst;
            if (sd.edgeOrg(e, &org) > 0 && sd.edgeDst(e, &dst) > 0) {
                drawLine(image, org, dst, red);
            }
            e = sd.getEdge(e, cv::Subdiv2D::NEXT_AROUND_LEFT);
            if (e == edge0) break;
        }
    }
    return sd.insert(p);
}

// Fill the polygon defined by points with a random color on image.
//
static void fillPoly(cv::Mat &image, const std::vector<cv::Point> &points)
{
    static const int lineKind = cv::LINE_8;
    static const int shift = 0;
    const cv::Scalar color = randomColor();
    cv::fillConvexPoly(image, points, color, lineKind, shift);
}

// Outline in black on image the closed polygon defined by points.
//
static void outlinePoly(cv::Mat &image, const std::vector<cv::Point> &points)
{
    static const bool isClosed = true;
    static const cv::Scalar black = cv::Scalar::all(0);
    static const int thickness = 1;
    static const int lineKind = cv::LINE_AA;
    static const int shift = 0;
    std::vector<std::vector<cv::Point> > polys(1, points);
    cv::polylines(image, polys, isClosed, black, thickness, lineKind, shift);
}

// Paint the Voronoi tesselation of sd in random colors on image.
// Draw edges in black and centers in white.
//
static void paintVoronoi(cv::Mat &image, cv::Subdiv2D &sd)
{
    static const cv::Scalar white = cv::Scalar::all(255);
    static const std::vector<int> noIndexes;
    std::vector<std::vector<cv::Point2f> > facets;
    std::vector<cv::Point2f> centers;
    sd.getVoronoiFacetList(noIndexes, facets, centers);
    for (std::size_t i = 0; i < facets.size(); ++i) {
        const std::vector<cv::Point2f> &f = facets[i];;
        std::vector<cv::Point> points(f.begin(), f.end());
        fillPoly(image, points);
        outlinePoly(image, points);
        drawPoint(image, centers[i], white);
    }
}

int main(int ac, const char *av[])
{
    if (ac == 1) {
        static const cv::Rect canvas(0, 0, 700, 800);
        static const cv::Mat blank = cv::Mat::zeros(canvas.size(), CV_8UC3);
        cv::Subdiv2D subdiv(canvas);
        cv::Mat delaunay = blank.clone();
        cv::Mat voronoi = blank.clone();
        makeWindow("Mesh Demo Delaunay", delaunay, 2);
        makeWindow("Mesh Demo Voronoi", voronoi);
        showKeys(av[0]);
        for (int i = 0; i < 500; ++i) {
            const int index = addRandomPoint(delaunay, subdiv);
            std::cout << "index == " << index << std::endl;
            cv::imshow("Mesh Demo Delaunay", delaunay);
            if (cv::waitKey(100) >= 0) break;
            blank.copyTo(delaunay);
            drawDelaunay(delaunay, subdiv);
            cv::imshow("Mesh Demo Delaunay", delaunay);
            paintVoronoi(voronoi, subdiv);
            drawDelaunay(voronoi, subdiv);
            cv::imshow("Mesh Demo Voronoi", voronoi);
            if (cv::waitKey(100) >= 0) break;
        }
        cv::waitKey(0);
        return 0;
    }
    showUsage(av[0]);
    return 1;
}
