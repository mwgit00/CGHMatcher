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

#ifndef GRADIENT_MATCHER_H_
#define GRADIENT_MATCHER_H_

#include "ghbase.h"


namespace ghalgo
{
    constexpr double ANG_STEP_MAX = 254.0;
    constexpr double ANG_STEP_MIN = 4.0;

    class GradientMatcher
    {
    public:

        GradientMatcher();
        virtual ~GradientMatcher();

        void init(
            const int kblur = 7,
            const int ksobel = 7,
            const double magthr = 0.2,
            const double angstep = 8.0);

        // This is the preprocessing step for the "classic" Generalized Hough algorithm.
        // Calculates Sobel derivatives of input grayscale image.  Converts to polar coordinates and
        // finds magnitude and angle (orientation).  Converts angle to integer with 4 to 254 steps.
        // Masks the pixels with gradient magnitudes above a threshold.
        void create_masked_gradient_orientation_img(const cv::Mat& rimg, cv::Mat& rmgo);

        // Initializes Generalized Hough table from grayscale image.
        // Default parameters are good starting point for doing object identification.
        void init_ghough_table_from_img(const cv::Mat& rimg);

        // Encodes gradients of input image and applies Generalized Hough transform
        void apply_ghough(const cv::Mat& rin, cv::Mat& rgrad, cv::Mat& rmatch);

        // Loads an image from a file, scales it, blurs it, and creates Generalized Hough table from it.
        // It uses the settings that were applied by the "init" method.
        // A reference to a blank image is passed in.  The loaded image is passed back.
        void load_template(
            cv::Mat& template_image,
            const std::string& rsfile,
            const double prescale = 1.0);

    public:

        int m_kpreblur;
        int m_ksobel;

        double m_magthr;
        double m_angstep;
        double m_max_votes;

        int m_loopstep;
        ghalgo::LookupTable m_ghtable;
    };
}

#endif // GRADIENT_MATCHER_H_
