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

#include "opencv2/imgproc.hpp"

namespace ghalgo
{
    // This class combines an image point and the number of votes it will contribute.
    // A point may get more than one vote.
    class PtVotes
    {
    public:
        PtVotes() : pt(0, 0), votes(0) {}
        PtVotes(const cv::Point& _pt, const uint16_t _v) : pt(_pt), votes(_v) {}
        virtual ~PtVotes() {}
    public:
        cv::Point pt;
        uint16_t votes;
    };


    // This is one value element for the Generalized Hough lookup table.
    // A key will be mapped to a value with an array of point-vote structures.
    class PtVotesArray
    {
    public:
        PtVotesArray() : ct(0), pt_votes(nullptr) {}
        virtual ~PtVotesArray() { clear(); }
        void clear() { ct = 0;  if (pt_votes) { delete[] pt_votes; } pt_votes = nullptr; }
    public:
        size_t ct;
        PtVotes * pt_votes;
    };


    // This is a Non-STL data structure for a Generalized Hough lookup table.
    // The lookup operation is just an array access.
    class LookupTable
    {
    public:
        LookupTable() : img_sz(0, 0), elem_ct(0), elems(nullptr) {}
        virtual ~LookupTable() { clear(); }

        void clear()
        {
            if (elems != nullptr)
            {
                for (size_t i = 0; i < elem_ct; i++) { elems[i].clear(); }
                delete[] elems;
            }
            img_sz = { 0, 0 };
            elems = nullptr;
            elem_ct = 0;
        }
    public:
        cv::Size img_sz;
        size_t elem_ct;
        PtVotesArray * elems;
    };


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
        const ghalgo::LookupTable& rtable)
    {
        rvotes = cv::Mat::zeros(rkeyimg.size(), E_VOTE_IMG_TYPE);
        for (int i = rtable.img_sz.height / 2; i < rkeyimg.rows - rtable.img_sz.height / 2; i++)
        {
            const T_KEY * pix = rkeyimg.ptr<T_KEY>(i);
            for (int j = rtable.img_sz.width / 2; j < rkeyimg.cols - rtable.img_sz.width / 2; j++)
            {
                // look up voting table for pixel
                // iterate through the points (if any) and add votes
                T_KEY uu = pix[j];
                T_pt_votes * pt_votes = rtable.elems[uu].pt_votes;
                const size_t ct = rtable.elems[uu].ct;
                for (size_t k = 0; k < ct; k++)
                {
                    const cv::Point& rp = pt_votes[k].pt;
                    int mx = (j + rp.x);
                    int my = (i + rp.y);
                    T_VOTE * pix = rvotes.ptr<T_VOTE>(my) + mx;
                    *pix += pt_votes[k].votes;
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
        const ghalgo::LookupTable& rtable)
    {
        rvotes = cv::Mat::zeros(rkeyimg.size(), E_VOTE_IMG_TYPE);
        for (int i = 1; i < (rkeyimg.rows - 1); i++)
        {
            const T_KEY * pix = rkeyimg.ptr<T_KEY>(i);
            for (int j = 1; j < (rkeyimg.cols - 1); j++)
            {
                // look up voting table for pixel
                // iterate through the points and add votes
                T_KEY uu = pix[j];
                PtVotes * pt_votes = rtable.elems[uu].pt_votes;
                const size_t ct = rtable.elems[uu].ct;
                for (size_t k = 0; k < ct; k++)
                {
                    // only vote if pixel is within output image bounds
                    const cv::Point& rp = pt_votes[k].pt;
                    int mx = (j + rp.x);
                    int my = (i + rp.y);
                    if ((mx >= 0) && (mx < rvotes.cols) &&
                        (my >= 0) && (my < rvotes.rows))
                    {
                        T_VOTE * pix = rvotes.ptr<T_VOTE>(my) + mx;
                        *pix += pt_votes[k].votes;
                    }
                }
            }
        }
    }
}

#endif // GHBASE_H_
