# CGHMatcher
More experiments with the Generalized Hough Transform.

This is a cleaned up implementation of the old BGHMatcher code.  The abstract Generalized Hough voting logic is all in a separate file now.  It uses STL containers for the voting logic.  It's slower in debug mode, but works fine in release mode.

There is a new "loop step" setting which can speed up the processing by skipping every 2nd, 3rd, or 4th row/column in the input image.  A by-product of the blurring steps is a lot of redundant votes.  Skipping rows/columns can still provide good results for large templates.

Template initialization has been reworked.  The test program has a new "acquisition" mode where a user can draw a rectangle with the mouse in the viewing window and then double-click to apply the image in the rectangle as the new template.  The test program also keeps track of the locations of the last several matches and displays their locations with small yellow diamonds.  There is also a new "feedback" mode where a smaller region centered in the current camera image becomes the new template to be matched against the next camera image.  This provides a way to detect motion.

The code has been tested with OpenCV 4.1.0 on a Windows 7 machine with Visual Studio 2015.  It has also been tested with OpenCV 4.3.0 on a Windows 10 machine with Visual Studio 2019.  I use the Community edition of Visual Studio.

Here is a demo video of the older BGHMatcher:

https://www.youtube.com/watch?v=heNQ9mr__L8

Here is a demo video of CGHMatcher showing some of the new features:

https://www.youtube.com/watch?v=1j2CDY2_c_k

