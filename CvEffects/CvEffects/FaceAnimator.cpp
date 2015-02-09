/*****************************************************************************
 *   FaceAnimator.cpp
 ******************************************************************************
 *   by Kirill Kornyakov and Alexander Shishkov, 13th May 2013
 ****************************************************************************** *
 *   Copyright Packt Publishing 2013.
 *   http://bit.ly/OpenCV_for_iOS_book
 *****************************************************************************/

#include "FaceAnimator.hpp"
#include "Processing.hpp"

#include "opencv2/imgproc/imgproc.hpp"

// Include the rest of our code!
#include "detectObject.h"       // Easily detect faces or eyes (using LBP or Haar Cascades).
#include "preprocessFace.h"     // Easily preprocess face images, for face recognition.
#include "recognition.h"     // Train the face recognition system and recognize a person from an image.

using namespace cv;
using namespace std;

// The Face Recognition algorithm can be one of these and perhaps more, depending on your version of OpenCV, which must be atleast v2.4.1:
//    "FaceRecognizer.Eigenfaces":  Eigenfaces, also referred to as PCA (Turk and Pentland, 1991).
//    "FaceRecognizer.Fisherfaces": Fisherfaces, also referred to as LDA (Belhumeur et al, 1997).
//    "FaceRecognizer.LBPH":        Local Binary Pattern Histograms (Ahonen et al, 2006).
//const char *facerecAlgorithm = "FaceRecognizer.Fisherfaces";
const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces";

// Sets how confident the Face Verification algorithm should be to decide if it is an unknown person or a known person.
// A value roughly around 0.5 seems OK for Eigenfaces or 0.7 for Fisherfaces, but you may want to adjust it for your
// conditions, and if you use a different Face Recognition algorithm.
// Note that a higher threshold value means accepting more faces as known people,
// whereas lower values mean more faces will be classified as "unknown".
const float UNKNOWN_PERSON_THRESHOLD = 0.7f;

enum MODES {MODE_STARTUP=0, MODE_DETECTION, MODE_COLLECT_FACES, MODE_TRAINING, MODE_RECOGNITION, MODE_DELETE_ALL,   MODE_END};
const char* MODE_NAMES[] = {"Startup", "Detection", "Collect Faces", "Training", "Recognition", "Delete All", "ERROR!"};
MODES m_mode = MODE_STARTUP;

int m_selectedPerson = -1;
int m_numPersons = 0;
vector<int> m_latestFaces;

// Set the desired face dimensions. Note that "getPreprocessedFace()" will return a square face.
const int faceWidth = 70;
const int faceHeight = faceWidth;

const bool preprocessLeftAndRightSeparately = true;   // Preprocess left & right sides of the face separately, in case there is stronger light on one side.

// Parameters controlling how often to keep new faces when collecting them. Otherwise, the training set could look to similar to each other!
const double CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3;      // How much the facial image should change before collecting a new face photo for training.
const double CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0;       // How much time must pass before collecting a new face photo for training.

const int BORDER = 12;  // Border between GUI elements to the edge of the image.

int m_gui_faces_left = -1;
int m_gui_faces_top = -1;

Ptr<FaceRecognizer> model;
vector<Mat> preprocessedFaces;
vector<int> faceLabels;
Mat old_prepreprocessedFace;
double old_time = 0;

void FaceAnimator::recognize()
{
    if (m_mode == MODE_COLLECT_FACES) {
        cout << "User wants to begin training." << endl;
        m_mode = MODE_TRAINING;
    }
}

void FaceAnimator::deleteAll()
{
    cout << "User clicked [Delete All] button." << endl;
    m_mode = MODE_DELETE_ALL;
}

void FaceAnimator::addPerson()
{
    // Check if the user clicked on one of our GUI buttons.
//    Point pt = Point(x,y);
//    if (isPointInRect(pt, m_rcBtnAdd)) {
        cout << "User clicked [Add Person] button when numPersons was " << m_numPersons << endl;
        // Check if there is already a person without any collected faces, then use that person instead.
        // This can be checked by seeing if an image exists in their "latest collected face".
        if ((m_numPersons == 0) || (m_latestFaces[m_numPersons-1] >= 0)) {
            // Add a new person.
            m_numPersons++;
            m_latestFaces.push_back(-1); // Allocate space for an extra person.
            cout << "Num Persons: " << m_numPersons << endl;
        }
        // Use the newly added person. Also use the newest person even if that person was empty.
        m_selectedPerson = m_numPersons - 1;
        m_mode = MODE_COLLECT_FACES;
//    }
//    else if (isPointInRect(pt, m_rcBtnDel)) {
//        cout << "User clicked [Delete All] button." << endl;
//        m_mode = MODE_DELETE_ALL;
//    }
//    else if (isPointInRect(pt, m_rcBtnDebug)) {
//        cout << "User clicked [Debug] button." << endl;
//        m_debug = !m_debug;
//        cout << "Debug mode: " << m_debug << endl;
//    }
//    else {
//        cout << "User clicked on the image" << endl;
//        // Check if the user clicked on one of the faces in the list.
//        int clickedPerson = -1;
//        for (int i=0; i<m_numPersons; i++) {
//            if (m_gui_faces_top >= 0) {
//                Rect rcFace = Rect(m_gui_faces_left, m_gui_faces_top + i * faceHeight, faceWidth, faceHeight);
//                if (isPointInRect(pt, rcFace)) {
//                    clickedPerson = i;
//                    break;
//                }
//            }
//        }
//        // Change the selected person, if the user clicked on a face in the GUI.
//        if (clickedPerson >= 0) {
//            // Change the current person, and collect more photos for them.
//            m_selectedPerson = clickedPerson; // Use the newly added person.
//            m_mode = MODE_COLLECT_FACES;
//        }
//        // Otherwise they clicked in the center.
//        else {
//            // Change to training mode if it was collecting faces.
//            if (m_mode == MODE_COLLECT_FACES) {
//                cout << "User wants to begin training." << endl;
//                m_mode = MODE_TRAINING;
//            }
//        }
//    }
}

// Draw text into an image. Defaults to top-left-justified text, but you can give negative x coords for right-justified text,
// and/or negative y coords for bottom-justified text.
// Returns the bounding rect around the drawn text.
Rect drawString(Mat img, string text, Point coord, Scalar color, float fontScale = 0.6f, int thickness = 1, int fontFace = FONT_HERSHEY_COMPLEX)
{
    // Get the text size & baseline.
    int baseline=0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;
    
    // Adjust the coords for left/right-justified or top/bottom-justified.
    if (coord.y >= 0) {
        // Coordinates are for the top-left corner of the text from the top-left of the image, so move down by one row.
        coord.y += textSize.height;
    }
    else {
        // Coordinates are for the bottom-left corner of the text from the bottom-left of the image, so come up from the bottom.
        coord.y += img.rows - baseline + 1;
    }
    // Become right-justified if desired.
    if (coord.x < 0) {
        coord.x += img.cols - textSize.width + 1;
    }
    
    // Get the bounding box around the text.
    Rect boundingRect = Rect(coord.x, coord.y - textSize.height, textSize.width, baseline + textSize.height);
    
    // Draw anti-aliased text.
    putText(img, text, coord, fontFace, fontScale, color, thickness, CV_AA);
    
    // Let the user know how big their text is, in case they want to arrange things.
    return boundingRect;
}

FaceAnimator::FaceAnimator(Parameters parameters)
{
    parameters_ = parameters;

    ExtractAlpha(parameters_.glasses, maskOrig_);
    ExtractAlpha(parameters_.mustache, maskMust_);
}

void FaceAnimator::putImage(Mat& frame, const Mat& image,
                            const Mat& alpha, Rect face,
                            Rect feature, float shift)
{
    // Scale animation image
    float scale = 1.1;
    Size size;
    size.width = scale * feature.width;
    size.height = scale * feature.height;
    Size newSz = Size(size.width,
                      float(image.rows) / image.cols * size.width);
    Mat glasses;
    Mat mask;
    resize(image, glasses, newSz);
    resize(alpha, mask, newSz);

    // Find place for animation
    float coeff = (scale - 1.) / 2.;
    Point origin(face.x + feature.x - coeff * feature.width,
                 face.y + feature.y - coeff * feature.height +
                 newSz.height * shift);
    Rect roi(origin, newSz);
    Mat roi4glass = frame(roi);
    
    alphaBlendC4(glasses, roi4glass, mask);
}

static bool FaceSizeComparer(const Rect& r1, const Rect& r2)
{
    return r1.area() > r2.area();
}

void FaceAnimator::PreprocessToGray(Mat& frame)
{
    cvtColor(frame, grayFrame_, CV_RGBA2GRAY);
    equalizeHist(grayFrame_, grayFrame_);
}

void FaceAnimator::PreprocessToGray_optimized(Mat& frame)
{
    grayFrame_.create(frame.size(), CV_8UC1);
    accBuffer1_.create(frame.size(), frame.type());
    accBuffer2_.create(frame.size(), CV_8UC1);
        
    cvtColor_Accelerate(frame, grayFrame_, accBuffer1_, accBuffer2_);
    equalizeHist_Accelerate(grayFrame_, grayFrame_);
}

void FaceAnimator::detectAndAnimateFaces(Mat& displayedFrame)
{
    if (m_mode == MODE_STARTUP)
    {
        // Since we have already initialized everything, lets start in Detection mode.
        m_mode = MODE_DETECTION;
    }
        // Run the face recognition system on the camera image. It will draw some things onto the given image, so make sure it is not read-only memory!
        int identity = -1;
        
        // Find a face and preprocess it to have a standard size and contrast & brightness.
        Rect faceRect;  // Position of detected face.
        Rect searchedLeftEye, searchedRightEye; // top-left and top-right regions of the face, where eyes were searched.
        Point leftEye, rightEye;    // Position of the detected eyes.
        Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, parameters_.faceCascade, parameters_.eyesCascade1, parameters_.eyesCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);
        
        bool gotFaceAndEyes = false;
        if (preprocessedFace.data)
            gotFaceAndEyes = true;
        
        // Draw an anti-aliased rectangle around the detected face.
        if (faceRect.width > 0) {
            rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);
            
            // Draw light-blue anti-aliased circles for the 2 eyes.
            Scalar eyeColor = CV_RGB(0,255,255);
            if (leftEye.x >= 0) {   // Check if the eye was detected
                circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 2, CV_AA);
            }
            if (rightEye.x >= 0) {   // Check if the eye was detected
                circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 2, CV_AA);
            }
        }

        if (m_mode == MODE_DETECTION) {
            // Don't do anything special.
        }
        else if (m_mode == MODE_COLLECT_FACES) {
            // Check if we have detected a face.
            if (gotFaceAndEyes) {
                
                // Check if this face looks somewhat different from the previously collected face.
                double imageDiff = 10000000000.0;
                if (old_prepreprocessedFace.data) {
                    imageDiff = getSimilarity(preprocessedFace, old_prepreprocessedFace);
                }
                
                // Also record when it happened.
                double current_time = (double)getTickCount();
                double timeDiff_seconds = (current_time - old_time)/getTickFrequency();
                
                // Only process the face if it is noticeably different from the previous frame and there has been noticeable time gap.
                if ((imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION) && (timeDiff_seconds > CHANGE_IN_SECONDS_FOR_COLLECTION)) {
                    // Also add the mirror image to the training set, so we have more training data, as well as to deal with faces looking to the left or right.
                    Mat mirroredFace;
                    flip(preprocessedFace, mirroredFace, 1);
                    
                    // Add the face images to the list of detected faces.
                    preprocessedFaces.push_back(preprocessedFace);
                    preprocessedFaces.push_back(mirroredFace);
                    faceLabels.push_back(m_selectedPerson);
                    faceLabels.push_back(m_selectedPerson);
                    
                    // Keep a reference to the latest face of each person.
                    m_latestFaces[m_selectedPerson] = preprocessedFaces.size() - 2;  // Point to the non-mirrored face.
                    // Show the number of collected faces. But since we also store mirrored faces, just show how many the user thinks they stored.
                    cout << "Saved face " << (preprocessedFaces.size()/2) << " for person " << m_selectedPerson << endl;
                    
                    // Make a white flash on the face, so the user knows a photo has been taken.
                    Mat displayedFaceRegion = displayedFrame(faceRect);
                    displayedFaceRegion += CV_RGB(90,90,90);
                    
                    // Keep a copy of the processed face, to compare on next iteration.
                    old_prepreprocessedFace = preprocessedFace;
                    old_time = current_time;
                }
            }
        }
        else if (m_mode == MODE_TRAINING) {
            
            // Check if there is enough data to train from. For Eigenfaces, we can learn just one person if we want, but for Fisherfaces,
            // we need atleast 2 people otherwise it will crash!
            bool haveEnoughData = true;
            if (strcmp(facerecAlgorithm, "FaceRecognizer.Fisherfaces") == 0) {
                if ((m_numPersons < 2) || (m_numPersons == 2 && m_latestFaces[1] < 0) ) {
                    cout << "Warning: Fisherfaces needs atleast 2 people, otherwise there is nothing to differentiate! Collect more data ..." << endl;
                    haveEnoughData = false;
                }
            }
            if (m_numPersons < 1 || preprocessedFaces.size() <= 0 || preprocessedFaces.size() != faceLabels.size()) {
                cout << "Warning: Need some training data before it can be learnt! Collect more data ..." << endl;
                haveEnoughData = false;
            }
            
            if (haveEnoughData) {
                // Start training from the collected faces using Eigenfaces or a similar algorithm.
                model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm);
                
                // Show the internal face recognition data, to help debugging.
//                if (m_debug)
//                    showTrainingDebugData(model, faceWidth, faceHeight);
                
                // Now that training is over, we can start recognizing!
                m_mode = MODE_RECOGNITION;
            }
            else {
                // Since there isn't enough training data, go back to the face collection mode!
                m_mode = MODE_COLLECT_FACES;
            }
            
        }
        else if (m_mode == MODE_RECOGNITION) {
            if (gotFaceAndEyes && (preprocessedFaces.size() > 0) && (preprocessedFaces.size() == faceLabels.size())) {
                
                // Generate a face approximation by back-projecting the eigenvectors & eigenvalues.
                Mat reconstructedFace;
                reconstructedFace = reconstructFace(model, preprocessedFace);
//                if (m_debug)
//                    if (reconstructedFace.data)
//                        imshow("reconstructedFace", reconstructedFace);
                
                // Verify whether the reconstructed face looks like the preprocessed face, otherwise it is probably an unknown person.
                double similarity = getSimilarity(preprocessedFace, reconstructedFace);
                
                string outputStr;
                if (similarity < UNKNOWN_PERSON_THRESHOLD) {
                    // Identify who the person is in the preprocessed face image.
                    identity = model->predict(preprocessedFace);
                    outputStr = to_string(identity);
                }
                else {
                    // Since the confidence is low, assume it is an unknown person.
                    outputStr = "Unknown";
                }
                cout << "Identity: " << outputStr << ". Similarity: " << similarity << endl;
                
                // Show the confidence rating for the recognition in the mid-top of the display.
                int cx = (displayedFrame.cols - faceWidth) / 2;
                Point ptBottomRight = Point(cx - 5, BORDER + faceHeight);
                Point ptTopLeft = Point(cx - 15, BORDER);
                // Draw a gray line showing the threshold for an "unknown" person.
                Point ptThreshold = Point(ptTopLeft.x, ptBottomRight.y - (1.0 - UNKNOWN_PERSON_THRESHOLD) * faceHeight);
                rectangle(displayedFrame, ptThreshold, Point(ptBottomRight.x, ptThreshold.y), CV_RGB(200,200,200), 1, CV_AA);
                // Crop the confidence rating between 0.0 to 1.0, to show in the bar.
                double confidenceRatio = 1.0 - min(max(similarity, 0.0), 1.0);
                Point ptConfidence = Point(ptTopLeft.x, ptBottomRight.y - confidenceRatio * faceHeight);
                // Show the light-blue confidence bar.
                rectangle(displayedFrame, ptConfidence, ptBottomRight, CV_RGB(0,255,255), CV_FILLED, CV_AA);
                // Show the gray border of the bar.
                rectangle(displayedFrame, ptTopLeft, ptBottomRight, CV_RGB(200,200,200), 1, CV_AA);
            }
        }
        else if (m_mode == MODE_DELETE_ALL) {
            // Restart everything!
            m_selectedPerson = -1;
            m_numPersons = 0;
            m_latestFaces.clear();
            preprocessedFaces.clear();
            faceLabels.clear();
            old_prepreprocessedFace = Mat();
            
            // Restart in Detection mode.
            m_mode = MODE_DETECTION;
        }
        else {
            cerr << "ERROR: Invalid run mode " << m_mode << endl;
            exit(1);
        }
        
        
        // Show the help, while also showing the number of collected faces. Since we also collect mirrored faces, we should just
        // tell the user how many faces they think we saved (ignoring the mirrored faces), hence divide by 2.
        string help;
        Rect rcHelp;
        if (m_mode == MODE_DETECTION)
            help = "Click [Add Person] when ready to collect faces.";
        else if (m_mode == MODE_COLLECT_FACES)
            help = "Click anywhere to train from your " + to_string(preprocessedFaces.size()/2) + " faces of " + to_string(m_numPersons) + " people.";
        else if (m_mode == MODE_TRAINING)
            help = "Please wait while your " + to_string(preprocessedFaces.size()/2) + " faces of " + to_string(m_numPersons) + " people builds.";
        else if (m_mode == MODE_RECOGNITION)
            help = "Click people on the right to add more faces to them, or [Add Person] for someone new.";
        if (help.length() > 0) {
            // Draw it with a black background and then again with a white foreground.
            // Since BORDER may be 0 and we need a negative position, subtract 2 from the border so it is always negative.
            float txtSize = 0.4;
            drawString(displayedFrame, help, Point(BORDER, -BORDER-2), CV_RGB(0,0,0), txtSize);  // Black shadow.
            rcHelp = drawString(displayedFrame, help, Point(BORDER+1, -BORDER-1), CV_RGB(255,255,255), txtSize);  // White text.
        }
        
        // Show the current mode.
        if (m_mode >= 0 && m_mode < MODE_END) {
            string modeStr = "MODE: " + string(MODE_NAMES[m_mode]);
            drawString(displayedFrame, modeStr, Point(BORDER, -BORDER-2 - rcHelp.height), CV_RGB(0,0,0));       // Black shadow
            drawString(displayedFrame, modeStr, Point(BORDER+1, -BORDER-1 - rcHelp.height), CV_RGB(0,255,0)); // Green text
        }
        
        // Show the current preprocessed face in the top-center of the display.
        int cx = (displayedFrame.cols - faceWidth) / 2;
        if (preprocessedFace.data) {
            // Get a BGR version of the face, since the output is BGR color.
            Mat srcBGR = Mat(preprocessedFace.size(), displayedFrame.type());
            cvtColor(preprocessedFace, srcBGR, CV_GRAY2RGBA);
            // Get the destination ROI (and make sure it is within the image!).
            //min(m_gui_faces_top + i * faceHeight, displayedFrame.rows - faceHeight);
            Rect dstRC = Rect(cx, BORDER, faceWidth, faceHeight);
            Mat dstROI = displayedFrame(dstRC);
            // Copy the pixels from src to dst.
            srcBGR.copyTo(dstROI);
        }
        // Draw an anti-aliased border around the face, even if it is not shown.
        rectangle(displayedFrame, Rect(cx-1, BORDER-1, faceWidth+2, faceHeight+2), CV_RGB(200,200,200), 1, CV_AA);
        
//        // Draw the GUI buttons into the main image.
//        m_rcBtnAdd = drawButton(displayedFrame, "Add Person", Point(BORDER, BORDER));
//        m_rcBtnDel = drawButton(displayedFrame, "Delete All", Point(m_rcBtnAdd.x, m_rcBtnAdd.y + m_rcBtnAdd.height), m_rcBtnAdd.width);
//        m_rcBtnDebug = drawButton(displayedFrame, "Debug", Point(m_rcBtnDel.x, m_rcBtnDel.y + m_rcBtnDel.height), m_rcBtnAdd.width);
    
        // Show the most recent face for each of the collected people, on the right side of the display.
        m_gui_faces_left = displayedFrame.cols - BORDER - faceWidth;
        m_gui_faces_top = BORDER;
        for (int i=0; i<m_numPersons; i++) {
            int index = m_latestFaces[i];
            if (index >= 0 && index < (int)preprocessedFaces.size()) {
                Mat srcGray = preprocessedFaces[index];
                if (srcGray.data) {
                    // Get a BGR version of the face, since the output is BGR color.
                    Mat srcBGR = Mat(srcGray.size(), displayedFrame.type());
                    cvtColor(srcGray, srcBGR, CV_GRAY2RGBA);
                    // Get the destination ROI (and make sure it is within the image!).
                    int y = min(m_gui_faces_top + i * faceHeight, displayedFrame.rows - faceHeight);
                    Rect dstRC = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
                    Mat dstROI = displayedFrame(dstRC);
                    // Copy the pixels from src to dst.
                    srcBGR.copyTo(displayedFrame(Rect(m_gui_faces_left, y, faceWidth, faceHeight)));
                }
            }
        }
        
        // Highlight the person being collected, using a red rectangle around their face.
        if (m_mode == MODE_COLLECT_FACES) {
            if (m_selectedPerson >= 0 && m_selectedPerson < m_numPersons) {
                int y = min(m_gui_faces_top + m_selectedPerson * faceHeight, displayedFrame.rows - faceHeight);
                Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
                rectangle(displayedFrame, rc, CV_RGB(255,0,0), 1, CV_AA);
            }
        }
    
        // Highlight the person that has been recognized, using a green rectangle around their face.
        if (identity >= 0 && identity < 1000) {
            int y = min(m_gui_faces_top + identity * faceHeight, displayedFrame.rows - faceHeight);
            Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
            rectangle(displayedFrame, rc, CV_RGB(0,255,0), 1, CV_AA);
        }
        
        // Show the camera frame on the screen.
//        imshow(windowName, displayedFrame);
    
        // If the user wants all the debug data, show it to them!
//        if (m_debug) {
//            Mat face;
//            if (faceRect.width > 0) {
//                face = cameraFrame(faceRect);
//                if (searchedLeftEye.width > 0 && searchedRightEye.width > 0) {
//                    Mat topLeftOfFace = face(searchedLeftEye);
//                    Mat topRightOfFace = face(searchedRightEye);
//                    imshow("topLeftOfFace", topLeftOfFace);
//                    imshow("topRightOfFace", topRightOfFace);
//                }
//            }
//            
//            if (!model.empty())
//                showTrainingDebugData(model, faceWidth, faceHeight);
//        }
    
        
        // IMPORTANT: Wait for atleast 20 milliseconds, so that the image can be displayed on the screen!
        // Also checks if a key was pressed in the GUI window. Note that it should be a "char" to support Linux.
//        char keypress = waitKey(20);  // This is needed if you want to see anything!
    
//        if (keypress == VK_ESCAPE) {   // Escape Key
//            // Quit the program!
//            break;
//        }
    
//    }//end while
    

//    TS(Preprocessing);
//    //PreprocessToGray(frame);
//    PreprocessToGray_optimized(frame);
//    TE(Preprocessing);
//    
//    // Detect faces
//    TS(DetectFaces);
//    std::vector<Rect> faces;
//    parameters_.faceCascade.detectMultiScale(grayFrame_, faces, 1.1,
//                                              2, 0, Size(100, 100));
//    TE(DetectFaces);
//    printf("Detected %lu faces\n", faces.size());
//
//    // Sort faces by size in descending order
//    sort(faces.begin(), faces.end(), FaceSizeComparer);
//
//    for ( size_t i = 0; i < faces.size(); i++ )
//    {
//        Mat faceROI = grayFrame_( faces[i] );
//
//        std::vector<Rect> facialFeature;
//        if (i % 2 == 0)
//        {// Detect eyes
//            Point origin(0, faces[i].height/4);
//            Mat eyesArea = faceROI(Rect(origin,
//                        Size(faces[i].width, faces[i].height/4)));
//
//            TS(DetectEyes);
//            parameters_.eyesCascade.detectMultiScale(eyesArea,
//                facialFeature, 1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT,
//                Size(faces[i].width * 0.55, faces[i].height * 0.13));
//            TE(DetectEyes);
//            
//            if (facialFeature.size())
//            {
//                TS(DrawEyes);
//                Scalar eyeColor = CV_RGB(0,255,255);
//                Point origin(faces[i].x + facialFeature[0].x * 1.13 + facialFeature[0].width/4,
//                             faces[i].y + facialFeature[0].y * 4.55 + facialFeature[0].height/2);
//                if (facialFeature[0].x >= 0) {   // Check if the eye was detected
//                    circle(frame, origin, 6, eyeColor, 2, CV_AA);
//                }
////                putImage(frame, parameters_.glasses, maskOrig_,
////                         faces[i], facialFeature[0] + origin, -0.1f);
//                TE(DrawEyes);
//            }
//        }
//        else
//        {// Detect mouth
//            Point origin(0, faces[i].height/2);
//            Mat mouthArea = faceROI(Rect(origin,
//                Size(faces[i].width, faces[i].height/2)));
//
//            parameters_.mouthCascade.detectMultiScale(
//                mouthArea, facialFeature, 1.1, 2,
//                CV_HAAR_FIND_BIGGEST_OBJECT,
//                Size(faces[i].width * 0.2, faces[i].height * 0.13) );
//            
//            if (facialFeature.size())
//            {
//                putImage(frame, parameters_.mustache, maskMust_,
//                         faces[i], facialFeature[0] + origin, 0.3f);
//            }
//        }
//    }
}
