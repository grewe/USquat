package com.edu.usquat.LookLearn;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.os.SystemClock;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.edu.usquat.LookLearn.env.ImageUtils;
import com.edu.usquat.LookLearn.env.Logger;
import com.edu.usquat.LookLearn.tflite.Classifier;
import com.edu.usquat.LookLearn.tflite.TFLiteObjectDetectionEfficientDet;
import com.edu.usquat.R;

import androidx.core.content.res.ResourcesCompat;




public class LookLearnProcessor {

    TFLiteObjectDetectionEfficientDet detector;
    Context context;
    /**
     * percent as a decimal (0 - 1.0) of the opacity factor used in the creation of
     * forced attention Look Learn images for the part deemed background
     */
    float backgroundPercent;
    /**
     * percent as a decimal (0 - 1.0) of the opacity factor used in the creation of
     * forced attention Look Learn images for the part deemed persons/bodys detected
     */
    float bodyPercent;
    /**
     * percent as a decimal (0 - 1.0) of the opacity factor used in the creation of
     * forced attention Look Learn images for the part deemed body parts (knees, feet, hips)
     */
    float bodyPartPercent;


    // Configuration values for the BodyPartCNN TFlite model
    private static final int TF_OD_API_INPUT_SIZE = 380;    //this is the wxh of square input size to MODEL
    private static final boolean TF_OD_API_IS_QUANTIZED = true;  //if its quantized or not. MUST be whatever the save tflite model is saved as

    //TFlite file for BodyPartCNN
    private static final String TF_OD_API_MODEL_FILE = "BodyPartCNN1.tflite"; //"IRdetect.tflite";   //name of input file for MODEL must be tflite format
    //LabelMap file listed classes--same order as training
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/BodyPartCNNLabelMap.txt";
    //private static final String TF_OD_API_MODEL_FILE = "DPDMdetector.tflite"; //"IRdetect.tflite";   //name of input file for MODEL must be tflite format
    //LabelMap file listed classes--same order as training
    //private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/DPDMlabelmap.txt";



    //???NEED not using Activity
    // private static final DetectorActivity.DetectorMode MODE = DetectorActivity.DetectorMode.TF_OD_API;   //Using Object Detection API

    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;   //a detected prediction must have value > threshold to be displayed
    private static final boolean MAINTAIN_ASPECT = false;  //if you want to keep aspect ration or not --THIS must be same as what is expected in model,done in training


    //cropsize for images before passing through the BodyPartCNN
    int cropSize;

    //Logger instance
    private static final Logger LOGGER = new Logger();

    //timestamp to track timing
    private long timestamp = 0;
    private long lastProcessingTimeMs;   //last time processed a frame


    //constructor that will get passed the Activity which own's it's context
    // reads in parameters used by LookLearn
    public LookLearnProcessor(Context c) {
        this.context = c;

        //retrieve the percentages used in the forced attention Look Learn image creation
        backgroundPercent = ResourcesCompat.getFloat(context.getResources(), R.dimen.backgroundPercent);
        bodyPercent = ResourcesCompat.getFloat(context.getResources(), R.dimen.bodyPercent);
        bodyPartPercent = ResourcesCompat.getFloat(context.getResources(), R.dimen.bodyPartPercent);


        //load up the LookLearn Forced Attention Detector
        //Loading the BodyPartCNN
        try {
            detector =
                    (TFLiteObjectDetectionEfficientDet) TFLiteObjectDetectionEfficientDet.create(
                            context.getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing BodyPartCNN!");
            Toast toast =
                    Toast.makeText(
                            context, "BodyPartCNN could not be initialized", Toast.LENGTH_SHORT);
            toast.show();

        }

    }

    /**
     * returns true if box1 is inside of box2
     * input_frames = list of image bitmaps
     * box1, box 2 are RectF and contain left, right,top,bottom.
     * NOTE: that the parameters of the RectF are floating points normalized between 0.0 and 1.0
     * as these represent bounding boxes from an Ojbect Detection model and that is the output (normalized) not
     * the raw pixel row and column values
     *
     * @return list of forced attention Look Learn Images
     */
    public Boolean boxInBox(RectF box1, RectF box2) {
        if(box1.left >=box2.left && box1.left <=box2.right &&  //left vertical edge of box 1 inside boxbox1.right <=box2.right &&  //the vertical edges of box 1 inside of box2
           box1.right >= box2.left && box1.right <=box2.right && //right vertical edge of box 1 inside box2
           box1.top >= box2.top && box1.top <= box2.bottom && //top horizontal edge of box 1 inside of box2
           box1.bottom >= box2.top && box1.bottom <=box2.bottom ) {   //the horizontal edges of box 1 inside of box2
            return Boolean.TRUE;
        }
        else
            return Boolean.FALSE;

    }

    /**
     *
     *  Retruns true if box1 is within a % (local variable fuzz-Pixs_normalized) of box2's bounding values
     *  box 1 and box2 are RectF with pixel locations represented from 0.0 to 1.0 which are normalized values
     *  spanning the entire scene/image width/height
     *    NOTE: that the parameters of the RectF are floating points normalized between 0.0 and 1.0
     *     as these represent bounding boxes from an Ojbect Detection model and that is the output (normalized) not
     *     the raw pixel row and column values
     *     NOTE:  fuzz_pix_normalized = 0.1 would mean within 10% of width/height of the original image size
     * @param box1
     * @param box2
     * @return
     */
    public Boolean fuzzyBoxInBox(RectF box1, RectF box2) {
        double fuzz_pixs_normalized = 0.1;
        if(box1.left >= box2.left - fuzz_pixs_normalized && box1.left <= box2.right + fuzz_pixs_normalized && // left vertical edge of box 1 is withing fuzz_pix_normalized% of box2 boundaries (or inside)
           box1.right >= box2.left - fuzz_pixs_normalized && box1.right <= box2.right + fuzz_pixs_normalized && // right vertical edge of box 1 is in range
           box1.top >= box2.top - fuzz_pixs_normalized && box2.top <= box2.bottom + fuzz_pixs_normalized && // top horizontal edge of box 1 in range
           box2.bottom >= box2.top - fuzz_pixs_normalized && box2.bottom <= box2.bottom + fuzz_pixs_normalized ) {  //bottom horizontal edge of box 1 in range
            return Boolean.TRUE;
        }
        else {
            return Boolean.FALSE;
        }

    }

    /**
     * box is RectF with bounds between 0.0 and 1.0 as it has been normalized by size of image in width/height
     * @return true rNormalized,cNormalized if in bounds if box
     */
    public  Boolean inBox(RectF box, float rNormalized, float cNormalized) {
        if (rNormalized <= box.bottom && rNormalized >= box.top && cNormalized <= box.right && cNormalized >= box.left)
            return Boolean.TRUE;
        else
            return Boolean.FALSE;
    }

    /**
     * box is representing a bounding box in image with normalized values between 0.0 and 1.0 for width and height
     * r,c is row,column of a pixel in original image coordinates
     * image_size is size of image ---ASSUMES images are square
     * So, factor is image_size when compare
     * @return
     */
    public boolean inBox(RectF box, int r, int c, int image_size) {
        if (r <= box.bottom * image_size && r >= box.top * image_size &&
                c <= box.right * image_size && c >= box.left * image_size)
            return Boolean.TRUE;
        else
            return Boolean.FALSE;
    }

    /**
     *
     * returns True if (rNormalized,cNormalized) is inside any of the boxes in the boxes[] array
     * Note Rect.left,right,top,bottom and rNormalized, cNormalized are normalized (0.0 to 1.0) row
     * and column locations relative to an image normalized so width,height range from 0.0 to 1.0
     *
     * @return true if rNormalized,cNormalized is a pixel location insize of one of the boxes in boxes[]
     */
    public  boolean inBoxes(ArrayList <RectF> boxes, float rNormalized, float cNormalized) {
        boolean inBoxesFlag = Boolean.FALSE;

        for(int i=0; i <= boxes.size(); i++){
            if (inBox(boxes.get(i), rNormalized, cNormalized))
                inBoxesFlag = Boolean.TRUE;
            break;
        }
        return inBoxesFlag;
    }


    /**
     * tell if
     * returns True if (r,c) is inside any of the boxes in the boxes[] array
     * Note Rect.left,right,top,bottom  are normalized (0.0 to 1.0) row and
     *  and column locations relative to an image normalized so width,height range from 0.0 to 1.0
     *
     *  Note (r,c) is not normalized, hence need image_size the size of image where assume image is square
     *
     * @return true if rNormalized,cNormalized is a pixel location insize of one of the boxes in boxes[]
     */
    public boolean inBoxes(ArrayList<RectF> boxes, int r, int c, int image_size){

         boolean inBoxesFlag = Boolean.TRUE;
        for(int i=0; i<= boxes.size(); i++){
            if (inBox(boxes.get(i), r, c, image_size))
                inBoxesFlag = Boolean.TRUE;
            break;
        }
        return inBoxesFlag;
}

    /**
     * method to generate a Bitmap representing the forced Attention Look&Learn image
     * this is an image where certain parts (bounding boxes) representing semantic regions of body and
     * body parts are emphasized by altering the "transparency" through darkening of the background,
     * body and body_parts by the this.**Percent multiplication factor.  (this.bodyPercent, this.backgroundPercent, etc)
     * @param input_frames
     * @return
     */
    public List<Bitmap> createAttentionImages(List<Bitmap> input_frames) {

        //NOTE--we will update the contents of input_frames with the new attention images if body is detection + parts


        //looping variables
        Classifier.Recognition result;
        String title;
        RectF box;


        int BODY_LABELS[] = {1,2};

        int BODY_PART_LABELS[] = {3,4,5};

        //body box  --initialize to nonsense
        RectF bodyBox = new RectF(-1,-1,-1,-1);

        //List of body part bounding boxes that will be inside of our selected bodyBox
        ArrayList<RectF> insideBodyPartBoxes = new ArrayList<RectF>();



        //saftey check
        if(input_frames.size() <=0)
            return input_frames;


        int IMG_SIZE = TF_OD_API_INPUT_SIZE;
        //get the image size from the first image in the input_frames
        int inputWidth = input_frames.get(0).getWidth();
        int inputHeight = input_frames.get(0).getHeight();

        ++timestamp;
        final long currTimestamp = timestamp;
        Bitmap output;
        int pixel;  //stored rgb as a combined value using Color.argb format
        int A,R,G,B;
        int i;


        //Now need to cycle through the frames in input_frames and run through the BodyPartCNN
        //first we need to resize it and then run it through the bodypartCNN
        for(int frame_index=0; frame_index<input_frames.size(); frame_index++) {

            //STEP 1: resize to run through body part CNN and create
            // LookLearn output Bitmap and visualize the background in it
            Bitmap b = Bitmap.createScaledBitmap(input_frames.get(frame_index), cropSize, cropSize, true);
            //make a copy to create our LookLearn image in
            output = b.copy(b.getConfig(), true);
            for (int x = 0; x < output.getWidth(); ++x) {
                for (int y = 0; y < output.getHeight(); ++y) {
                    // get pixel color
                    pixel = output.getPixel(x, y);
                    // apply filtering on each channel R, G, B
                    A = Color.alpha(pixel);
                    R = (int) (Color.red(pixel) * backgroundPercent);
                    G = (int) (Color.green(pixel) * backgroundPercent);
                    B = (int) (Color.blue(pixel) * backgroundPercent);
                    // set new color pixel to output bitmap
                    output.setPixel(x, y, Color.argb(A, R, G, B));
                }
            }


            //STEP2: run through BodyPartCNN and get detection
            // now start detections
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(b);  //performing detection on croppedBitmap
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            LOGGER.i("BodyPartCNN took to process 1 image ", lastProcessingTimeMs);
            LOGGER.i("BodyPartCNN took to process 1 image " + lastProcessingTimeMs);

            //STEP3: create the forced Attension image stored in b that uses the detections
            // SEE PYTHON CODE -- ANKUSH we did not do ANY and ALL recognitions above the threshold --only one body, etc.
            //cycling through all of the recognition detections in my image I am currently processing


            //STEP3.1:  find the best Body box --highest certainty
            // sort through and find the best body detection above teh min_score_threshol
            for (i = 0; i < results.size(); i++) {
                result = results.get(i);
                title = result.getTitle();
                if ((title.contains("bodySquat") || title.contains("bodyTall") || title.contains("Person")) && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                    bodyBox = result.getLocation();
                    break;
                }

            }

            //Saftey check --if no body found then continue the loop to process the next image
            if (bodyBox.left == -1)
                //do nothing to this input frame....go to next for processing
                continue;

            //STEP 3.2: Find BodyParts that are inside the bodyBox and save in a List<Classifier.Recognition>
            for (i = 0; i < results.size(); i++) {
                result = results.get(i);
                title = result.getTitle();
                box = result.getLocation();
                if (result.getConfidence() < MINIMUM_CONFIDENCE_TF_OD_API || title.contains("bodyTall") || title.contains("bodySquat") || title.contains("Person"))
                    continue;
                //test if bounding box of this body part is inside of our bodyBox we selected in STEP1
                if (fuzzyBoxInBox(box, bodyBox)) {
                    insideBodyPartBoxes.add(box);
                }
            }


            //STEP 3.3: Create Visualization of the bodyBox and the insideBodyPartBoxes in the current ouput image
            //step3.3.1:  first lest visualize the bodyBox
            int row, col;  //represents row and column
            //assumes image is square IMG_SIZExIMG_SIZE
           // for (row = (int) (bodyBox.top * IMG_SIZE); row <= (int) (bodyBox.bottom * IMG_SIZE); row++)
            //    for (col = (int) (bodyBox.left * IMG_SIZE); col <= (int) (bodyBox.right * IMG_SIZE); col++) {

            //saftey check
            if(bodyBox.top <0.0)
                bodyBox.top=0.0f;
            if(bodyBox.top >=IMG_SIZE)
                bodyBox.top = IMG_SIZE -1;
            if(bodyBox.left <0.0)
                bodyBox.left=0.0f;
            if(bodyBox.left >=IMG_SIZE)
                bodyBox.left = IMG_SIZE -1;
            if(bodyBox.right <0.0)
                bodyBox.right=0.0f;
            if(bodyBox.right >=IMG_SIZE)
                bodyBox.right = IMG_SIZE -1;
            if(bodyBox.bottom <0.0)
                bodyBox.bottom=0.0f;
            if(bodyBox.bottom >=IMG_SIZE)
                bodyBox.bottom = IMG_SIZE -1;

            for (row = (int) (bodyBox.top ); row <= (int) (bodyBox.bottom); row++)
                for (col = (int) (bodyBox.left); col <= (int) (bodyBox.right); col++) {
                    // get pixel color
                    pixel = b.getPixel(row, col);
                    // apply filtering on each channel R, G, B
                    A = Color.alpha(pixel);
                    R = (int) (Color.red(pixel) * bodyPercent);
                    G = (int) (Color.green(pixel) * bodyPercent);
                    B = (int) (Color.blue(pixel) * bodyPercent);
                    // set new color pixel to output bitmap
                    output.setPixel(row, col, Color.argb(A, R, G, B));
                }

            //step 3.3.2: lets visualize all of the insideBodyPartBoxes
            for(int index=0; index< insideBodyPartBoxes.size(); index++){
                //saftey check
                if(insideBodyPartBoxes.get(index).top <0.0)
                    insideBodyPartBoxes.get(index).top=0.0f;
                if(insideBodyPartBoxes.get(index).top >=IMG_SIZE)
                    insideBodyPartBoxes.get(index).top = IMG_SIZE -1;
                if(insideBodyPartBoxes.get(index).left <0.0)
                    insideBodyPartBoxes.get(index).left=0.0f;
                if(insideBodyPartBoxes.get(index).left >=IMG_SIZE)
                    insideBodyPartBoxes.get(index).left = IMG_SIZE -1;
                if(insideBodyPartBoxes.get(index).right <0.0)
                    insideBodyPartBoxes.get(index).right=0.0f;
                if(insideBodyPartBoxes.get(index).right >=IMG_SIZE)
                    insideBodyPartBoxes.get(index).right = IMG_SIZE -1;
                if(insideBodyPartBoxes.get(index).bottom <0.0)
                    insideBodyPartBoxes.get(index).bottom=0.0f;
                if(insideBodyPartBoxes.get(index).bottom >=IMG_SIZE)
                    insideBodyPartBoxes.get(index).bottom = IMG_SIZE -1;
                for (row = (int) (insideBodyPartBoxes.get(index).top * IMG_SIZE); row <= (int) (insideBodyPartBoxes.get(index).bottom * IMG_SIZE); row++)
                    for (col = (int) (insideBodyPartBoxes.get(index).left * IMG_SIZE); col <= (int) (insideBodyPartBoxes.get(index).right * IMG_SIZE); col++) {
                        // get pixel color
                        pixel = b.getPixel(row, col);
                        // apply filtering on each channel R, G, B
                        A = Color.alpha(pixel);
                        R = (int) (Color.red(pixel) * bodyPartPercent);
                        G = (int) (Color.green(pixel) * bodyPartPercent);
                        B = (int) (Color.blue(pixel) * bodyPartPercent);
                        // set new color pixel to output bitmap
                        output.setPixel(row, col, Color.argb(A, R, G, B));
                    }
            }





/*
                        //saftey check
                        int left = (int) location.left;
                        int right = (int) location.right;
                        int top = (int) location.top;
                        int bottom = (int) location.bottom;
                        if(left <0 )
                            left =0;
                        if(left>b.getWidth())
                            left = b.getWidth();
                        if(right <0 )
                            right =0;
                        if(right>b.getWidth())
                            right = b.getWidth();
                        if(left > right)
                            left=right;
                        if(top <0 )
                            top =0;
                        if(top>b.getHeight())
                            top = b.getHeight();
                        if(bottom <0 )
                            bottom =0;
                        if(bottom>b.getHeight())
                            bottom = b.getHeight();
                        if(top > bottom)
                            top=bottom;


                        //https://developer.android.com/reference/android/graphics/RectF
                    //cycle through entire image
                    //NOW create the image using bodyPart and insideBodyPartBoxes

                    for (int  x= left; x < right; ++x) {


                        for (int y = top; y< bottom ; ++y) {
                            if (inBoxes(insideBodyPartBoxes, x, y, IMG_SIZE)) {
                                pixel = b.getPixel(x, y);
                                // apply filtering on each channel R, G, B
                                A = Color.alpha(pixel);
                                R = (int) (Color.red(pixel) * bodyPartPercent);
                                G = (int) (Color.green(pixel) * bodyPartPercent);
                                B = (int) (Color.blue(pixel) * bodyPartPercent);
                            }
                            else if(inBox(bodyBox, x, y, IMG_SIZE)) {
                                pixel = b.getPixel(x, y);
                                // apply filtering on each channel R, G, B
                                A = Color.alpha(pixel);
                                R = (int) (Color.red(pixel) * bodyPercent);
                                G = (int) (Color.green(pixel) * bodyPercent);
                                B = (int) (Color.blue(pixel) * bodyPercent);
                            }
                            else {
                                pixel = b.getPixel(x, y);
                                // apply filtering on each channel R, G, B
                                A = Color.alpha(pixel);
                                R = (int) (Color.red(pixel) * backgroundPercent);
                                G = (int) (Color.green(pixel) * backgroundPercent);
                                B = (int) (Color.blue(pixel) * backgroundPercent);
                            }
                        }
                    }

*/


            //set the LookLearn image to replace the original
            input_frames.set(frame_index, output);





    }//end processing all of the frames

    return input_frames;
  }//end of create** method


}



