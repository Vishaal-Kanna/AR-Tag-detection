# AR Tag Detection
This project aims at detecting and identifying AR Tags from a given video sequence. Once identified we calculate homography matrices and overlay an image and project a virtual cube over the tag. Custom functions were written for AR-tag detection including finding contours, detecting the id and for image warping. 

## To run the code, follow the below steps:
1. Download and extract the .zip file
2. Copy video files and the reference tag into this folder or change the path in the Code_project1.py
3. Open the terminal from folder. Run python3 Code_project1.py.
4. The output is displayed with the Testudo image overlayed as well as the virtual cube on the Ar-Tag for 100ms per frame.
5. The image is displayed as the output without the cube or Testudo template if corners are not detected using the Corner detection pipeline
6. Link to the video - https://drive.google.com/file/d/1O1pmJqg19doau3KTLPreotUBibZd-TkY/view
