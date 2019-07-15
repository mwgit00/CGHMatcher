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
        const double angstep)
    {
        m_kblur = kblur;
        m_ksobel = ksobel;
        m_magthr = magthr;
        m_angstep = angstep;
        
        m_max_votes = 0.0;
        m_ghtable.clear();
    }


    void GradientMatcher::create_ghough_table(const cv::Mat& rgrad)
    {
        // key is 8-bit
        // max key is angle steps + 1 because both 0 and 2pi can come from polar conversion
        ghalgo::create_lookup_table(rgrad, static_cast<uint8_t>(m_angstep + 1.0), m_ghtable);
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


    void GradientMatcher::init_ghough_table_from_img(cv::Mat& rimg)
    {
        // create image of encoded Sobel gradient orientations from input image
        // then create Generalized Hough lookup table from that image
        cv::Mat img_cgrad;
        create_masked_gradient_orientation_img(rimg, img_cgrad);
        create_ghough_table(img_cgrad);
    }


    void GradientMatcher::apply_ghough(cv::Mat& rin, cv::Mat& rgrad, cv::Mat& rmatch)
    {
        // create image of encoded Sobel gradient orientations from input image
        // then apply Generalized Hough transform
        create_masked_gradient_orientation_img(rin, rgrad);
        apply_ghough_transform_allpix<uint8_t, CV_16U, uint16_t>(rgrad, rmatch, m_ghtable);
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

        // apply the pre-blur setting since processing loop also does a pre-blur
        GaussianBlur(scaled_template_image, scaled_template_image, { m_kblur, m_kblur }, 0);

        init_ghough_table_from_img(scaled_template_image);

        // after generating table, run transform on the scaled template image to get ideal max votes
        cv::Mat img_cgrad;
        cv::Mat img_match;
        apply_ghough(scaled_template_image, img_cgrad, img_match);
        minMaxLoc(img_match, nullptr, &m_max_votes, nullptr, nullptr);
    }
}
