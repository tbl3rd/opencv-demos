#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>

#include <iostream>


static void showUsage(const char *av0)
{
    std::cerr << av0 << ": Demo video background removal."
              << std::endl << std::endl
              << "Usage: " << av0 << " <camera> <output>" << std::endl
              << std::endl
              << "Where: <camera> is a camera number or video file name."
              << std::endl
              << "       <output> is where to write the modified video."
              << std::endl << std::endl
              << "Example: " << av0 << " 0 ./output.avi"
              << std::endl << std::endl;
}


// Create a new unobscured named window for image.
// Reset windows layout with when reset is not 0.
//
// The 23 term works around how MacOSX decorates windows.
//
static void makeWindow(const char *window, const cv::Size &size, int reset = 0)
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
    moveX += size.width;
    maxY = std::max(maxY, size.height);
}


// Just cv::VideoCapture extended for convenience.  The const_cast<>()s
// work around the missing member const on cv::VideoCapture::get().
//
struct CvVideoCapture: cv::VideoCapture {

    double getFramesPerSecond() const {
        CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
        const double fps = p->get(cv::CAP_PROP_FPS);
        return fps ? fps : 30.0;        // for MacBook iSight camera
    }

    int getFourCcCodec() const {
        CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
        return p->get(cv::CAP_PROP_FOURCC);
    }

    std::string getFourCcCodecString() const {
        char result[] = "????";
        CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
        const int code = p->getFourCcCodec();
        result[0] = ((code >>  0) & 0xff);
        result[1] = ((code >>  8) & 0xff);
        result[2] = ((code >> 16) & 0xff);
        result[3] = ((code >> 24) & 0xff);
        result[4] = ""[0];
        return std::string(result);
    }

    int getFrameCount() const {
        CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
        return p->get(cv::CAP_PROP_FRAME_COUNT);
    }

    cv::Size getFrameSize() const {
        CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
        const int w = p->get(cv::CAP_PROP_FRAME_WIDTH);
        const int h = p->get(cv::CAP_PROP_FRAME_HEIGHT);
        const cv::Size result(w, h);
        return result;
    }

    int getPosition(void) const {
        CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
        return p->get(cv::CAP_PROP_POS_FRAMES);
    }
    void setPosition(int p) { this->set(cv::CAP_PROP_POS_FRAMES, p); }

    CvVideoCapture(const std::string &fileName): VideoCapture(fileName) {}
    CvVideoCapture(int n): VideoCapture(n) {}
    CvVideoCapture(): VideoCapture() {}
};

// Return a video capture object suitable for the source string.
// Return the camera with specified ID if source contains an integer.
// Otherwise attempt to open a video file.
// Otherwise return the default camera (-1).
//
static CvVideoCapture openVideo(const char *source)
{
    int cameraId = 0;
    std::istringstream iss(source); iss >> cameraId;
    if (iss) return CvVideoCapture(cameraId);
    std::string filename;
    std::istringstream sss(source); sss >> filename;
    if (sss) return CvVideoCapture(filename);
    return CvVideoCapture(-1);
}


// Remove video background with BackgroundSubtractor classes.
//
template <typename PtrBs> class BackgroundRemover {
    PtrBs itsBs;
    cv::Mat itsMask;
    cv::Mat itsOutput;

    static PtrBs makeBs() { return PtrBs(0); }

public:

    // Apply frame to background tracker.
    //
    const cv::Mat &operator()(const cv::Mat &frame)
    {
        static cv::Mat black = cv::Mat::zeros(frame.size(), frame.type());
        itsBs->apply(frame, itsMask);
        black.copyTo(itsOutput);
        frame.copyTo(itsOutput, itsMask);
        return itsOutput;
    }

    BackgroundRemover(): itsBs(makeBs()) {}
};

// Hide differing create.*() syntax behind a function template.
//
template<> cv::Ptr<cv::BackgroundSubtractorGMG>
BackgroundRemover<cv::Ptr<cv::BackgroundSubtractorGMG> >::makeBs()
{
    return cv::Ptr<cv::BackgroundSubtractorGMG>
        (cv::createBackgroundSubtractorGMG());
}
template<> cv::Ptr<cv::BackgroundSubtractorMOG>
BackgroundRemover<cv::Ptr<cv::BackgroundSubtractorMOG> >::makeBs()
{
    return cv::Ptr<cv::BackgroundSubtractorMOG>
        (cv::createBackgroundSubtractorMOG());
}
template<> cv::Ptr<cv::BackgroundSubtractorMOG2>
BackgroundRemover<cv::Ptr<cv::BackgroundSubtractorMOG2> >::makeBs()
{
    return cv::Ptr<cv::BackgroundSubtractorMOG2>
        (cv::createBackgroundSubtractorMOG2());
}


// Hide template syntax behind typedefs.
//
typedef BackgroundRemover<cv::Ptr<cv::BackgroundSubtractorMOG> >
BackgroundRemoverMog;
typedef BackgroundRemover<cv::Ptr<cv::BackgroundSubtractorMOG2> >
BackgroundRemoverMog2;
typedef BackgroundRemover<cv::Ptr<cv::BackgroundSubtractorGMG> >
BackgroundRemoverGmg;


int main(int ac, const char *av[])
{
    if (ac == 3) {
        std::cout << av[0] << ": Camera is " << av[1] << std::endl;
        std::cout << av[0] << ": Output is " << av[2] << std::endl;
        CvVideoCapture camera = openVideo(av[1]);
        if (camera.isOpened()) {
            const int codec = camera.getFourCcCodec();
            const double fps = camera.getFramesPerSecond();
            const cv::Size size = camera.getFrameSize();
            const int count = camera.getFrameCount();
            cv::VideoWriter output(av[2], codec, fps, size);
            if (output.isOpened()) {
                std::cout << av[0] << ": " << camera.getFourCcCodecString()
                          << " " << count
                          << " (" << size.width << "x" << size.height << ")"
                          << " frames at " << fps << " FPS" << std::endl;
                std::cout << av[0] << ": Writing to " << av[2] << std::endl;
                static BackgroundRemoverMog br;
                for (int i = 0; i < count; ++i) {
                    static cv::Mat frame; camera >> frame;
                    if (!frame.empty()) output << br(frame);
                }
                return 0;
            }
        }
    }
    showUsage(av[0]);
    return 1;
}
