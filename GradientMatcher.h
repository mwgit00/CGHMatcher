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
            const double scale = 1.0,
            const double magthr = 1.0,
            const double angstep = 8.0);

        // This is the preprocessing step for the "classic" Generalized Hough algorithm.
        // Calculates Sobel derivatives of input grayscale image.  Converts to polar coordinates and
        // finds magnitude and angle (orientation).  Converts angle to integer with 4 to 254 steps.
        // Masks the pixels with gradient magnitudes above a threshold.
        void create_masked_gradient_orientation_img(
            const cv::Mat& rimg,
            cv::Mat& rmgo);

        // Creates a Generalized Hough lookup table from encoded gradient input image (CV_8U).
        // The scale parameter shrinks or expands the point set.
        void create_ghough_table(const cv::Mat& rgrad, const double scale);

        // Helper function for initializing Generalized Hough table from grayscale image.
        // Default parameters are good starting point for doing object identification.
        // Table must be a newly created object with blank data.
        void init_ghough_table_from_img(cv::Mat& rimg);

    public:

        int m_kblur;
        int m_ksobel;
        double m_scale;
        double m_magthr;
        double m_angstep;
        ghbase::T_lookup_table m_ghtable;
    };
}

#endif // GRADIENT_MATCHER_H_