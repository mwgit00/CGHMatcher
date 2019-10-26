// MIT License
//
// Copyright(c) 2019 Mark Whitney
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "Windows.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

#include "GradientMatcher.h"
#include "Knobs.h"
#include "util.h"


#define MATCH_DISPLAY_THRESHOLD (0.9)           // arbitrary
#define MOVIE_PATH              ".\\movie\\"    // user may need to create or change this
#define DATA_PATH               ".\\data\\"     // user may need to change this


using namespace cv;


#define SCA_BLACK   (cv::Scalar(0,0,0))
#define SCA_RED     (cv::Scalar(0,0,255))
#define SCA_GREEN   (cv::Scalar(0,255,0))
#define SCA_BLUE    (cv::Scalar(255,0,0))
#define SCA_MAGENTA (cv::Scalar(255,0,255))
#define SCA_YELLOW  (cv::Scalar(0,255,255))
#define SCA_WHITE   (cv::Scalar(255,255,255))


class MouseInfo
{
public:
    enum
    {
        MOFF = 0,
        MPT0 = 1,
        MMOV = 2,
        MPT1 = 3,
        MACQ = 4,
    };
    cv::Point pt0;
    cv::Point pt1;
    cv::Rect rect;
    int mstate;
    
    MouseInfo() { clear(); }
    virtual ~MouseInfo() {}
    void apply(bool x)
    {
        if (x)
        {
            if (mstate == MOFF) mstate = MPT0;
        }
        else
        {
            if (mstate != MOFF) clear();
        }
    }
    void clear() { pt0 = cv::Point(); pt1 = cv::Point(); rect = cv::Rect(); mstate = MOFF; }
};


Mat template_image;
Size g_viewer_size;
MouseInfo g_mouse_info;

ghalgo::GradientMatcher theMatcher;
const char * stitle = "CGHMatcher";
const double default_mag_thr = 0.2;
int n_record_ctr = 0;
size_t nfile = 0;


const std::vector<T_file_info> vfiles =
{
    { default_mag_thr, 1.5, "circle_b_on_w.png" },
    { default_mag_thr, 1.5, "ring_b_on_w.png" },
    { default_mag_thr, 3.0, "bottle_20perc_top_b_on_w.png" },
    { default_mag_thr, 3.5, "panda_face.png" },
    { default_mag_thr, 3.0, "stars_main.png" }
};


static bool wait_and_check_keys(Knobs& rknobs)
{
    bool result = true;

    int nkey = waitKey(1);
    char ckey = static_cast<char>(nkey);

    // check that a keypress has been returned
    if (nkey >= 0)
    {
        if (ckey == 27)
        {
            // done if ESC has been pressed
            result = false;
        }
        else
        {
            rknobs.handle_keypress(ckey);
        }
    }

    return result;
}


static void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    MouseInfo * pmi = reinterpret_cast<MouseInfo *>(userdata);
    if (pmi->mstate)
    {
        if (event == EVENT_LBUTTONDOWN)
        {
            if (pmi->mstate == MouseInfo::MPT0)
            {
                pmi->pt0 = Point(x, y);
                pmi->pt1 = Point(x, y);
                pmi->rect = Rect(pmi->pt0, pmi->pt1);
                pmi->mstate = MouseInfo::MMOV;
            }
        }
        else if (event == EVENT_LBUTTONUP)
        {
            if (pmi->mstate == MouseInfo::MMOV)
            {
                pmi->pt1 = Point(x, y);
                pmi->rect = Rect(pmi->pt0, pmi->pt1);
                pmi->mstate = MouseInfo::MPT1;
            }
        }
        else if (event == EVENT_MOUSEMOVE)
        {
            if (pmi->mstate == MouseInfo::MMOV)
            {
                pmi->pt1 = Point(x, y);
                pmi->rect = Rect(pmi->pt0, pmi->pt1);
            }
        }
        else if (event = EVENT_LBUTTONDBLCLK)
        {
            if (pmi->mstate == MouseInfo::MPT1)
            {
                pmi->mstate = MouseInfo::MACQ;
            }
        }
    }
}


static void image_output(
    Mat& rimg,
    const double qmax,
    const Point& rptmax,
    const Knobs& rknobs)
{
    if (rknobs.get_acq_mode_enabled())
    {
        // draw rectangle for acquisition region
        // and a blue box around entire screen
        rectangle(rimg, g_mouse_info.rect, SCA_GREEN, 3);
        rectangle(rimg, Rect(0, 0, rimg.size().width, rimg.size().height), SCA_BLUE, 3);
    }
    else
    {
        const int h_score = 16;
        const int w_score = 40;

        if (rknobs.get_template_display_enabled())
        {
            // draw current template in upper right corner
            Mat bgr_template_img;
            cvtColor(template_image, bgr_template_img, COLOR_GRAY2BGR);
            Size osz = rimg.size();
            Size tsz = template_image.size();
            Rect roi = cv::Rect(osz.width - tsz.width, 0, tsz.width, tsz.height);
            bgr_template_img.copyTo(rimg(roi));

            // draw colored box around template image (magenta if recording)
            cv::Scalar box_color = (rknobs.get_record_enabled()) ? SCA_MAGENTA : SCA_BLUE;
            rectangle(rimg, { osz.width - tsz.width, 0 }, { osz.width, tsz.height }, box_color, 2);
        }

        // determine size of "target" box
        Size rsz = theMatcher.m_ghtable.img_sz;
        Point corner = { rptmax.x - rsz.width / 2, rptmax.y - rsz.height / 2 };

        // a loop step of 2 means 1/4 of the pixels will be processed, 3 means 1/9 will be processed, etc.
        // so the score can be adjusted by the squared loop step to keep it consistent for different step values
        double step_scale = static_cast<double>(theMatcher.m_loopstep * theMatcher.m_loopstep);

        // format score string for viewer (#.##)
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << ((qmax / theMatcher.m_max_votes) * step_scale);

        // draw black background box then draw text score on top of it
        // dispaly location is adjusted based on visible corners (default is upper left)
        int score_y = (corner.y > h_score) ? (corner.y - h_score) : (corner.y + rsz.height);
        int score_x = (corner.x > 0) ? corner.x : corner.x + rsz.width - w_score;
        rectangle(rimg, { score_x, score_y, 40, h_score }, SCA_BLACK, -1);
        putText(rimg, oss.str(), { score_x, score_y + h_score - 4 }, FONT_HERSHEY_PLAIN, 1.0, SCA_WHITE, 1);

        // draw rectangle around best match with yellow dot at center
        rectangle(rimg, Rect(corner.x, corner.y, rsz.width, rsz.height), SCA_GREEN, 2);
        circle(rimg, rptmax, 2, SCA_YELLOW, -1);
    }

    // save each frame to a file if recording
    if (rknobs.get_record_enabled())
    {
        std::ostringstream osx;
        osx << MOVIE_PATH << "img_" << std::setfill('0') << std::setw(5) << n_record_ctr << ".png";
        imwrite(osx.str(), rimg);
        n_record_ctr++;
    }

    cv::imshow(stitle, rimg);
}


static void reload_template(
    const Knobs& rknobs,
    const T_file_info& rinfo)
{
    // with more "knobs" the magnitude threshold and angle step setting could also be re-applied here
    // but right now only the pre-blur Gaussian kernel size and Sobel kernel size can be adjusted on the fly
    std::string spath = DATA_PATH + rinfo.sname;
    theMatcher.init(rknobs.get_pre_blur(), rknobs.get_ksobel(), rinfo.mag_thr);
    theMatcher.load_template(template_image, spath, rinfo.img_scale);
    std::cout << "LOADED:  blur=" << rknobs.get_pre_blur() << ", sobel=" << rknobs.get_ksobel();
    std::cout << ", magthr=" << rinfo.mag_thr << ", " << rinfo.sname << " ";
    std::cout << theMatcher.m_max_votes << std::endl;
}


static void pre_process(Knobs& rknobs, Ptr<CLAHE>& rpCLAHE, cv::Mat& rimg_cam, cv::Mat& rimg_gray)
{
    // apply the current channel setting
    int nchan = rknobs.get_channel();
    if (nchan == Knobs::ALL_CHANNELS)
    {
        // combine all channels into grayscale
        cvtColor(rimg_cam, rimg_gray, COLOR_BGR2GRAY);
    }
    else
    {
        // select only one BGR channel
        Mat img_channels[3];
        split(rimg_cam, img_channels);
        rimg_gray = img_channels[nchan];
    }

    // apply the current histogram equalization setting
    if (rknobs.get_equ_hist_enabled())
    {
        double c = rknobs.get_clip_limit();
        rpCLAHE->setClipLimit(c);
        rpCLAHE->apply(rimg_gray, rimg_gray);
    }

    // apply the current blur setting
    int kpreblur = rknobs.get_pre_blur();
    if (kpreblur > 1)
    {
        GaussianBlur(rimg_gray, rimg_gray, { kpreblur, kpreblur }, 0);
    }
}


static void loop(void)
{
    Knobs theKnobs;
    int op_id;

    double qmax;
    Size capture_size;
    Point ptmax;

    Mat img;
    Mat img_viewer;
    Mat img_gray;
    Mat img_grad;
    Mat img_match;

    // set up mouse callback
    namedWindow(stitle);
    setMouseCallback(stitle, CallBackFunc, &g_mouse_info);

    // create a histogram equalizer
    Ptr<CLAHE> pCLAHE = createCLAHE();

    // need a 0 as argument for the video capture thing
    VideoCapture vcap(0);
    if (!vcap.isOpened())
    {
        std::cout << "Failed to open VideoCapture device!" << std::endl;
        ///////
        return;
        ///////
    }
    
    // camera is ready so grab a first image to determine its full size
    vcap >> img;
    capture_size = img.size();

    // use dummy operation to print initial Knobs settings message
    // and force template to be loaded at start of loop
    theKnobs.handle_keypress('0');

    // initialize lookup table
    reload_template(theKnobs, vfiles[nfile]);

    // and the image processing loop is running...
    bool is_running = true;

    while (is_running)
    {
        // grab image
        vcap >> img;

        // apply the current image scale setting
        double img_scale = theKnobs.get_img_scale();
        g_viewer_size = Size(
            static_cast<int>(capture_size.width * img_scale),
            static_cast<int>(capture_size.height * img_scale));
        resize(img, img_viewer, g_viewer_size);

        pre_process(theKnobs, pCLAHE, img_viewer, img_gray);

        // check for any operations that
        // might halt or reset the image processing loop
        if (theKnobs.get_op_flag(op_id))
        {
            if (op_id == Knobs::OP_TEMPLATE || op_id == Knobs::OP_UPDATE)
            {
                // changing the template will advance the file index
                if (op_id == Knobs::OP_TEMPLATE)
                {
                    nfile = (nfile + 1) % vfiles.size();
                }
                reload_template(theKnobs, vfiles[nfile]);
            }
            else if (op_id == Knobs::OP_RECORD)
            {
                if (theKnobs.get_record_enabled())
                {
                    // reset recording frame counter
                    std::cout << "RECORDING STARTED" << std::endl;
                    n_record_ctr = 0;
                }
                else
                {
                    std::cout << "RECORDING STOPPED" << std::endl;
                }
            }
            else if (op_id == Knobs::OP_MAKE_VIDEO)
            {
                std::cout << "CREATING VIDEO FILE..." << std::endl;
                std::list<std::string> listOfPNG;
                get_dir_list(MOVIE_PATH, "*.jpg", listOfPNG);
                bool is_ok = make_video(5.0, MOVIE_PATH,
                    "movie.mov",
                    VideoWriter::fourcc('M', 'P', '4', 'V'),
                    listOfPNG);
                std::cout << ((is_ok) ? "SUCCESS!" : "FAILURE!") << std::endl;
            }
        }

        if (g_mouse_info.mstate == MouseInfo::MACQ)
        {
            // use the PRE-PROCESSED image in the acquisition rectangle as the new template
            // apply the current Sobel filter size since this is used directly in the gradient calc
            Mat acq_img = img_gray(g_mouse_info.rect);
            theMatcher.m_ksobel = theKnobs.get_ksobel();
            theMatcher.m_magthr = default_mag_thr;
            theMatcher.init_ghough_table_from_img(acq_img);
            acq_img.copyTo(template_image);
            theKnobs.toggle_acq_mode_enabled();
            g_mouse_info.apply(false);
            std::cout << "New template acquired from camera" << std::endl;
        }

        // set loop iteration step
        // this will skip points in the input image for significant speed-up
        // then apply Generalized Hough transform and locate maximum (best match)
        theMatcher.m_loopstep = theKnobs.get_loopstep();
        theMatcher.apply_ghough(img_gray, img_grad, img_match);
        minMaxLoc(img_match, nullptr, &qmax, nullptr, &ptmax);

        // apply the current output mode
        // content varies but all final output images are BGR
        int nmode = theKnobs.get_output_mode();
        g_mouse_info.apply(theKnobs.get_acq_mode_enabled());
        if (theKnobs.get_acq_mode_enabled())
        {
            nmode = Knobs::OUT_COLOR;
        }
        
        switch (nmode)
        {
            case Knobs::OUT_RAW:
            {
                // show the raw match result
                Mat temp_8U;
                normalize(img_match, img_match, 0, 255, cv::NORM_MINMAX);
                img_match.convertTo(temp_8U, CV_8U);
                cvtColor(temp_8U, img_viewer, COLOR_GRAY2BGR);
                break;
            }
            case Knobs::OUT_GRAD:
            {
                // display encoded gradient image
                // show red overlay of any matches that exceed arbitrary threshold
                Mat match_mask;
                std::vector<std::vector<cv::Point>> contours;
                normalize(img_grad, img_grad, 0, 255, cv::NORM_MINMAX);
                cvtColor(img_grad, img_viewer, COLOR_GRAY2BGR);
                normalize(img_match, img_match, 0, 1, cv::NORM_MINMAX);
                match_mask = (img_match > MATCH_DISPLAY_THRESHOLD);
                findContours(match_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                drawContours(img_viewer, contours, -1, SCA_RED, -1, LINE_8, noArray(), INT_MAX);
                break;
            }
            case Knobs::OUT_PREP:
            {
                cvtColor(img_gray, img_viewer, COLOR_GRAY2BGR);
                break;
            }
            case Knobs::OUT_COLOR:
            default:
            {
                // no extra output processing
                break;
            }
        }

        // always show best match contour and target dot on BGR image
        image_output(img_viewer, qmax, ptmax, theKnobs);

        // handle keyboard events and end when ESC is pressed
        is_running = wait_and_check_keys(theKnobs);
    }

    // when everything is done, release the capture device and windows
    vcap.release();
    destroyAllWindows();
}


int main(int argc, char** argv)
{
    loop();
    return 0;
}
