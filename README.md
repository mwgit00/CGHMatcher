# CGHMatcher
More experiments with the Generalized Hough Transform.

This is a cleaned up implementation of the old BGHMatcher code.  The abstract Generalized Hough voting logic is all in a separate file now.  It uses STL containers for the voting logic.  It's slower in debug mode, but works fine in release mode.

There is a new "loop step" setting which can speed up the processing by skipping every 2nd, 3rd, or 4th row/column in the input image.  A by-product of the blurring steps is a lot of redundant votes.  Skipping rows/columns can still provide good results for large templates.

Template initialization has been reworked.  The test program has a new "acquisition" mode where a user can draw a rectangle with the mouse in the viewing window and then double-click to apply the image in the rectangle as the new template.

The code has been tested with OpenCV 4.1.0 on a Windows 7 machine with Visual Studio 2015 and a Windows 10 machine with Visual Studio 2019.  I use the Community edition of Visual Studio.
