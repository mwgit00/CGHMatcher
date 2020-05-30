// MIT License
//
// Copyright(c) 2020 Mark Whitney
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

#include "opencv2/highgui.hpp"
#include "GradientMatcher.h"


namespace ghalgo
{
    GradientMatcher::GradientMatcher()
    {
        init();
    }


    GradientMatcher::~GradientMatcher()
    {
    }


    void GradientMatcher::init(
        const int kblur,
        const int ksobel,
        const double magthr,
        const double angstep,
        const bool is_pre_CLAHE_enabled,
        const int CLAHE_clip_limit)
    {
        // parameters for generating a template
        m_kpreblur = kblur;
        m_ksobel = ksobel;
        m_magthr = magthr;
        m_angstep = angstep;
        m_is_pre_CLAHE_enabled = is_pre_CLAHE_enabled;
        m_CLAHE_clip_limit = CLAHE_clip_limit;
        
        m_max_votes = 0.0;
        m_loopstep = 1;
        m_ghtable.clear();
    }


    void GradientMatcher::create_masked_gradient_orientation_img(
        const cv::Mat& rimg,
        cv::Mat& rmgo)
    {
        double qmax;
        double angstep = m_angstep;
        cv::Mat temp_dx;
        cv::Mat temp_dy;
        cv::Mat temp_mag;
        cv::Mat temp_ang;
        cv::Mat temp_mask;
        const int SOBEL_DEPTH = CV_32F;

        // calculate X and Y gradients for input image
        cv::Sobel(rimg, temp_dx, SOBEL_DEPTH, 1, 0, m_ksobel);
        cv::Sobel(rimg, temp_dy, SOBEL_DEPTH, 0, 1, m_ksobel);

        // convert X-Y gradients to magnitude and angle
        cartToPolar(temp_dx, temp_dy, temp_mag, temp_ang);

        // create mask for pixels that exceed gradient magnitude threshold
        minMaxLoc(temp_mag, nullptr, &qmax);
        temp_mask = (temp_mag > (qmax * m_magthr));

        // scale, offset, and convert the angle image so 0-2pi becomes integers 1 to (ANG_STEP+1)
        // note that the angle can sometimes be 2pi which is equivalent to an angle of 0
        // for some binary source images not all gradient codes may be generated
        angstep = (angstep > ANG_STEP_MAX) ? ANG_STEP_MAX : angstep;
        angstep = (angstep < ANG_STEP_MIN) ? ANG_STEP_MIN: angstep;
        temp_ang.convertTo(rmgo, CV_8U, angstep / (CV_2PI), 1.0);

        // apply mask to eliminate pixels
        rmgo &= temp_mask;
    }


    void GradientMatcher::init_ghough_table_from_img(const cv::Mat& rimg)
    {
        cv::Mat img_cgrad;

        // create image of encoded Sobel gradient orientations from input image
        // then create Generalized Hough lookup table from that image
        create_masked_gradient_orientation_img(rimg, img_cgrad);

        // key is 8-bit
        // max key is angle steps + 1 because both 0 and 2pi can come from polar conversion
        // the 0 and 2pi values are equivalent but it's one extra "key" that must be handled
        ghalgo::create_lookup_table(img_cgrad, static_cast<uint8_t>(m_angstep + 1.0), m_ghtable);

        // stash floating point value of ideal max votes
        m_max_votes = static_cast<double>(m_ghtable.max_votes);
    }


    void GradientMatcher::apply_ghough(const cv::Mat& rin, cv::Mat& rgrad, cv::Mat& rmatch)
    {
        // create image of encoded Sobel gradient orientations from input image
        // then apply Generalized Hough transform
        create_masked_gradient_orientation_img(rin, rgrad);
        apply_ghough_transform_allpix<uint8_t, CV_16U, uint16_t>(rgrad, rmatch, m_ghtable, m_loopstep);
    }


    void GradientMatcher::load_template(
        cv::Mat& template_image,
        const std::string& rsfile,
        const double prescale)
    {
        template_image = cv::imread(rsfile, cv::IMREAD_GRAYSCALE);

        // scale the template image prior to generating table
        // use recommended interpolation method when shrinking or enlarging
        cv::Mat scaled_template_image;
        resize(template_image, scaled_template_image, cv::Size(), prescale, prescale, (prescale > 1.0) ? cv::INTER_CUBIC : cv::INTER_AREA);

        // GH pipeline should do the following:
        // get gray image -> perform optional histogram equalization -> perform pre-blur -> do GH

        // apply the optional histogram equalization setting
        if (m_is_pre_CLAHE_enabled)
        {
            cv::Ptr<cv::CLAHE> pCLAHE = cv::createCLAHE();
            pCLAHE->setClipLimit(static_cast<double>(m_CLAHE_clip_limit));
            pCLAHE->apply(scaled_template_image, scaled_template_image);
        }

        // apply the pre-blur setting
        if (m_kpreblur > 1)
        {
            GaussianBlur(scaled_template_image, scaled_template_image, { m_kpreblur, m_kpreblur }, 0);
        }

        // now that image has been pre-processed according to steps above
        // use it to generate the lookup table
        init_ghough_table_from_img(scaled_template_image);
    }
}
