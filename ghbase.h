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

#ifndef GHBASE_H_
#define GHBASE_H_

#include <vector>
#include "opencv2/imgproc.hpp"

namespace ghalgo
{
    class LookupTable
    {
    public:
        LookupTable() { clear(); }
        virtual ~LookupTable() {}
        void clear()
        {
            max_votes = 0;
            img_sz = cv::Size(0, 0);
            elems.clear();
        }
    public:
        size_t max_votes;
        cv::Size img_sz;
        std::vector< std::vector<cv::Point> > elems;
    };

    
    template<typename T_KEY>
    void create_lookup_table(const cv::Mat& rkey, const T_KEY max_key, ghalgo::LookupTable& rtable)
    {
        // calculate centering offset
        int row_offset = rkey.rows / 2;
        int col_offset = rkey.cols / 2;

        // the size of the table is max_key + 1 since it contains keys 0 to max_key
        size_t iimax = max_key + 1;

        // blow away old table
        // reserve space for table entries except for key 0 which is unused
        rtable.clear();
        rtable.elems.resize(iimax);
        for (size_t ii = 1; ii < iimax; ++ii)
        {
            rtable.elems[ii].reserve(2048);
        }

        // iterate through the key image pixel-by-pixel
        rtable.img_sz = rkey.size();
        for (int i = 0; i < rkey.rows; ++i)
        {
            const T_KEY * pix = rkey.ptr<T_KEY>(i);
            for (int j = 0; j < rkey.cols; ++j)
            {
                // process everything with non-zero key
                const T_KEY ghkey = pix[j];
                if (ghkey)
                {
                    // the vote count is mapped to a point and incremented
                    // max possible votes is number of non-zero keys
                    cv::Point offset_pt = cv::Point(col_offset - j, row_offset - i);
                    rtable.elems[ghkey].push_back(offset_pt);
                    rtable.max_votes++;
                }
            }
        }
    }


    // Applies Generalized Hough transform to an encoded "key" image.
    // The key should be a type suitable for an array index:  CV_8U or CV_16U.
    // Template parameters specify key type and output image type.  Examples:
    // - uint8_t key and votes are float:       <uint8_t,CV_32F,float>
    // - uint16_t key and votes are uint16_t:   <uint16_t,CV_16U,uint16_t>
    // The size of the target image used to generate the table will constrain the results.
    // Pixels near border and within half the X or Y dimensions of target image will be 0.
    // Output image is same size as input.  Maxima indicate good matches.
    template<typename T_KEY, int E_VOTE_IMG_TYPE, typename T_VOTE>
    void apply_ghough_transform(
        const cv::Mat& rkeyimg,
        cv::Mat& rvotes,
        const ghalgo::LookupTable& rtable,
        const int ijstep = 1)
    {
        rvotes = cv::Mat::zeros(rkeyimg.size(), E_VOTE_IMG_TYPE);
        for (int i = rtable.img_sz.height / 2; i < rkeyimg.rows - rtable.img_sz.height / 2; i += ijstep)
        {
            const T_KEY * pix = rkeyimg.ptr<T_KEY>(i);
            for (int j = rtable.img_sz.width / 2; j < rkeyimg.cols - rtable.img_sz.width / 2; j += ijstep)
            {
                // look up voting table for key
                // iterate through the points (if any) and add votes
                T_KEY uu = pix[j];
                const size_t ct = rtable.elems[uu].size();
                for (size_t k = 0; k < ct; ++k)
                {
                    const cv::Point& rp = rtable.elems[uu][k];
                    const int mx = (j + rp.x);
                    const int my = (i + rp.y);
                    T_VOTE * pix = rvotes.ptr<T_VOTE>(my) + mx;
                    T_VOTE& votes = *pix;
                    votes++;
                }
            }
        }
    }


    // Applies Generalized Hough transform to an encoded "key" image.
    // The key should be a type suitable for an array index:  CV_8U or CV_16U.
    // Template parameters specify key type and output image type.  Examples:
    // - uint8_t key and votes are float:       <uint8_t,CV_32F,float>
    // - uint16_t key and votes are uint16_t:   <uint16_t,CV_16U,uint16_t>
    // Each vote is range-checked.  Votes that would fall outside the image are discarded.
    // Output image is same size as input.  Maxima indicate good matches.
    template<typename T_KEY, int E_VOTE_IMG_TYPE, typename T_VOTE>
    void apply_ghough_transform_allpix(
        const cv::Mat& rkeyimg,
        cv::Mat& rvotes,
        const ghalgo::LookupTable& rtable,
        const int ijstep = 1)
    {
        rvotes = cv::Mat::zeros(rkeyimg.size(), E_VOTE_IMG_TYPE);
        for (int i = 1; i < (rkeyimg.rows - 1); i += ijstep)
        {
            const T_KEY * pix = rkeyimg.ptr<T_KEY>(i);
            for (int j = 1; j < (rkeyimg.cols - 1); j += ijstep)
            {
                // look up points associated with key
                // iterate through the points and add votes
                const T_KEY uu = pix[j];
                const size_t ct = rtable.elems[uu].size();
                for (size_t k = 0; k < ct; ++k)
                {
                    // only vote if pixel is within output image bounds
                    const cv::Point& rp = rtable.elems[uu][k];
                    const int mx = (j + rp.x);
                    const int my = (i + rp.y);
                    if ((mx >= 0) && (mx < rvotes.cols) &&
                        (my >= 0) && (my < rvotes.rows))
                    {
                        T_VOTE * pix = rvotes.ptr<T_VOTE>(my) + mx;
                        T_VOTE& votes = *pix;
                        votes++;
                    }
                }
            }
        }
    }
}

#endif // GHBASE_H_
