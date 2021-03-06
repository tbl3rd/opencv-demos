#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>

#include <iostream>


// Show the hot-keys on os.
//
static void showKeys(std::ostream &os, const char *av0)
{
    os << std::endl
       << av0 << ": Use keys to modify tracking behavior and display."
       << std::endl << std::endl
       << av0 << ": q to quit the program." << std::endl
       << av0 << ": t to find good tracking points." << std::endl
       << av0 << ": c to clear all tracking points." << std::endl
       << av0 << ": n to toggle the backing video display." << std::endl
       << std::endl
       << av0 << ": Click the mouse to add a tracking point." << std::endl
       << std::endl
       << av0 << ": If you are playing a video file ..." << std::endl
       << av0 << ": s to step the video by a frame." << std::endl
       << av0 << ": r to run the video at speed." << std::endl
       << std::endl;
}

// Show a usage message for av0 on stderr.
//
static void showUsage(const char *av0)
{
    std::cerr
        << av0
        << ": Demonstrate optical flow tracking after background removal."
        << std::endl << std::endl
        << "Usage: " << av0 << " <video>" << std::endl << std::endl
        << "Where: <video> is an optional video file." << std::endl
        << "       If <video> is '-' use a camera instead." << std::endl
        << std::endl
        << "Example: " << av0 << " - # use a camera" << std::endl
        << "Example: " << av0 << " ../resources/Megamind.avi"
        << std::endl << std::endl;
    showKeys(std::cerr, av0);
}

// Return termination criteria suitable for this program.
//
static cv::TermCriteria makeTerminationCriteria(void)
{
    static const int criteria
        = cv::TermCriteria::COUNT | cv::TermCriteria::EPS;
    static const int iterations = 20;
    static const double epsilon = 0.03;
    return cv::TermCriteria(criteria, iterations, epsilon);
}


// Just cv::VideoCapture extended for the convenience of
// FkltVideoPlayer.  The const_cast<>()s work around
// the missing member const on cv::VideoCapture::get().
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


// Play video from file with title at FPS or by stepping frames using a
// trackbar as a scrub control.
//
class FkltVideoPlayer {

    CvVideoCapture video;           // the video in this player
    std::string title;              // the title of the player window
    const int msDelay;              // the frame delay in milliseconds
    const int frameCount;           // 0 or number of frames in video
    cv::Mat image;                  // the output image in title window
    cv::Mat itsFrame;               // buffer of current frame from video
    cv::Mat priorGray;              // prior frame in grayscale
    cv::Mat nextGray;               // current frame in grayscale

    std::vector<cv::Point2f> priorPoints; // tracking points in priorGray
    std::vector<cv::Point2f> nextPoints;  // tracking points in nextGray

    int position;                   // 0 or current frame position in video
    enum State { RUN, STEP } state; // run at FPS or step frame by frame
    bool night;                     // true for no backing video in image

    // NONE  means no user hot-key request is pending
    // POINT means newPoint contains a new tracking point from mouse
    // CLEAR means points should be cleared on next frame
    // TRACK means automatically find good tracking points in next frame
    //
    enum Mode { NONE, POINT, CLEAR, TRACK } mode;

    cv::Point2f newPoint;                 // new point from mouse

    BackgroundRemoverMog itsBackgroundRemover;

    cv::CascadeClassifier    itsBodyHaar;

    // Draw a filled green circle of radius 3 on image at center.
    //
    static void drawGreenCircle(cv::Mat &image, const cv::Point &center)
    {
        static const int radius = 3;
        static const cv::Scalar green(0, 255, 0);
        static const int thickness = cv::FILLED;
        static const int lineKind = cv::LINE_8;
        cv::circle(image, center, radius, green, thickness, lineKind);
    }

    // Called by setMouseCallback() to add new points to track.
    //
    static void onMouseClick(int event, int x, int y, int n, void *p)
    {
        FkltVideoPlayer *const pV = (FkltVideoPlayer *)p;
        if (event == cv::EVENT_LBUTTONDOWN) {
            pV->newPoint = cv::Point2f(x, y);
            pV->mode = FkltVideoPlayer::POINT;
        }
    }

    // Return up to count good tracking points in gray.
    //
    static std::vector<cv::Point2f> getGoodTrackingPoints(const cv::Mat &gray)
    {
        static const int count = 500;
        static const double quality = 0.01;
        static const double minDistance = 10;
        static const cv::Mat noMask;
        static const int blockSize = 3;
        static const bool useHarrisDetector = false;
        static const double k = 0.04;
        std::vector<cv::Point2f> result;
        cv::goodFeaturesToTrack(gray, result, count, quality, minDistance,
                                noMask, blockSize, useHarrisDetector, k);
        std::cerr << "getGoodTrackingPoints(): " << result.size() << std::endl;
        return result;
    }

    // Calculate the flow of priorPoints in priorGray into nextPoints in
    // nextGray.
    //
    void drawFlowPoints(void)
    {
        if (!priorPoints.empty()) {
            static const cv::Size winSize(31, 31);
            static const int level = 3;
            static const cv::TermCriteria termCrit = makeTerminationCriteria();
            static const int flags = 0;
            static const double eigenThreshold = 0.001;
            std::vector<uchar> status;
            std::vector<float> error;
            cv::calcOpticalFlowPyrLK(priorGray, nextGray,
                                     priorPoints, nextPoints,
                                     status, error, winSize, level,
                                     termCrit, flags, eigenThreshold);
            nextPoints = drawPoints(image, status, nextPoints);
        }
    }

    // Draw on image each point from points whose status is true and return
    // all the points drawn.
    //
    static std::vector<cv::Point2f>
    drawPoints(cv::Mat &image,
               const std::vector<uchar> &status,
               const std::vector<cv::Point2f> &points)
    {
        int good = 0;
        std::vector<cv::Point2f> result;
        const int count = points.size();
        for (int i = 0; i < count; ++i) {
            if (status[i]) {
                ++good;
                const cv::Point2f point = points[i];
                result.push_back(point);
                drawGreenCircle(image, point);
            }
        }
        std::cerr << "drawPoints(): " << good << " / " << count << std::endl;
        return result;
    }

    // Add newPoint to points, after adjusting it to the nearest good
    // corner in gray.  Return the adjusted new point.
    //
    static cv::Point2f addTrackingPoint(std::vector<cv::Point2f> &points,
                                        const cv::Mat &gray,
                                        const cv::Point2f newPoint)
    {
        static const cv::Size winSize(31, 31);
        static const cv::Size noZeroZone(-1, -1);
        static const cv::TermCriteria termCrit = makeTerminationCriteria();
        std::vector<cv::Point2f> vnp;
        vnp.push_back(newPoint);
        cv::cornerSubPix(gray, vnp, winSize, noZeroZone, termCrit);
        const cv::Point2f result = vnp[0];
        points.push_back(result);
        return result;
    }

    // Adjust image for night and mode settings, then track and draw points
    // on image.
    //
    void handleModes(void)
    {
        if (night) image = cv::Scalar::all(0);
        if (0 == position % 16) mode = TRACK;
        if (mode == CLEAR) {
            priorPoints.clear();
            nextPoints.clear();
        } else if (mode == TRACK) {
            priorPoints = getGoodTrackingPoints(priorGray);
        } else if (mode == POINT) {
            const cv::Point2f p
                = addTrackingPoint(nextPoints, nextGray, newPoint);
            drawGreenCircle(image, p);
        }
        mode = NONE;
    }


    // Use a new frame unrelated to the prior frame.
    //
    void reset(void)
    {
        const int p = video.getPosition();
        video >> itsFrame;
        video.setPosition(p);
        cv::cvtColor(itsBackgroundRemover(itsFrame), nextGray,
                     cv::COLOR_BGR2GRAY);
        nextGray.copyTo(priorGray);
    }

    // Show the frame at position updating trackbar state as necessary.
    // Handle any mode set by hot-key and save prior state for later use.
    //
    void showFrame(void) {
        video >> itsFrame;
        if (itsFrame.data) {
            if (frameCount) {
                position = video.getPosition();
                cv::setTrackbarPos("Position", title, position);
            }
            itsFrame.copyTo(image);
            cv::cvtColor(itsBackgroundRemover(itsFrame), nextGray,
                         cv::COLOR_BGR2GRAY);
            handleModes();
            drawFlowPoints();
            std::swap(priorPoints, nextPoints);
            std::swap(priorGray, nextGray);
            cv::imshow(title, image);
        } else {
            state = STEP;
        }
    }

    // This is the trackbar callback where p is this FkltVideoPlayer.
    //
    static void onTrackBar(int position, void *p)
    {
        FkltVideoPlayer *const pV = (FkltVideoPlayer *)p;
        pV->video.setPosition(position);
        pV->state = FkltVideoPlayer::STEP;
        pV->reset();
        pV->showFrame();
    }

    friend std::ostream &operator<<(std::ostream &os, const FkltVideoPlayer &p)
    {
        const CvVideoCapture &v = p.video;
        const cv::Size s = v.getFrameSize();
        const int count = v.getFrameCount();
        if (count) os << count << " ";
        os << "(" << s.width << "x" << s.height << ") frames of ";
        if (count) os << v.getFourCcCodecString() << " ";
        os <<"video at " << v.getFramesPerSecond() << " FPS";
        return os;
    }

public:

    ~FkltVideoPlayer() { cv::destroyWindow(title); }

    // True if this can play.
    //
    operator bool() const { return video.isOpened(); }

    // Analyze the video frame-by-frame according to hot-key commands.
    // Return true unless something goes wrong.
    //
    bool operator()(void) {
        while (*this) {
            showFrame();
            const int wait = state == RUN ? msDelay : 0;
            const char c = cv::waitKey(wait);
            switch (c) {
            case 'q': case 'Q': return true;
            case 'n': case 'N': night = !night; break;
            case 't': case 'T': mode  = TRACK;  break;
            case 'c': case 'C': mode  = CLEAR;  break;
            case 'r': case 'R': state = RUN;    break;
            case 's': case 'S': state = STEP;   break;
            }
        }
        return false;
    }

    // Run Lukas-Kanade tracking on video from file t.
    //
    FkltVideoPlayer(const char *t):
        video(t), title(t), msDelay(1000 / video.getFramesPerSecond()),
        frameCount(video.getFrameCount()),
        position(0), state(RUN), night(false)
    {
        if (*this) {
            cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
            cv::setMouseCallback(title, &onMouseClick, this);
            cv::createTrackbar("Position", title, &position, frameCount,
                               &onTrackBar, this);
            reset();
        }
    }

    // Run Lukas-Kanade tracking on video from camera n.
    //
    FkltVideoPlayer(int n):
        video(n), title("Camera "), msDelay(1000 / video.getFramesPerSecond()),
        frameCount(0), position(0), state(RUN), night(false)
    {
        if (*this) {
            title += std::to_string(n);
            cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
            cv::setMouseCallback(title, &onMouseClick, this);
            reset();
        }
    }
};


int main(int ac, const char *av[])
{
    if (ac == 2) {
        if (0 == strcmp(av[1], "-")) {
            FkltVideoPlayer camera(-1);
            if (camera) showKeys(std::cout, av[0]);
            if (camera) std::cout << camera << std::endl;
            if (camera()) return 0;
        } else {
            FkltVideoPlayer video(av[1]);
            if (video) showKeys(std::cout, av[0]);
            if (video) std::cout << video << std::endl;
            if (video()) return 0;
        }
    }
    showUsage(av[0]);
    return 1;
}
