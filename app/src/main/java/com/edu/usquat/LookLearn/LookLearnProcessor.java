package com.edu.usquat.LookLearn;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.os.SystemClock;
import android.widget.Toast;

import java.io.IOException;
import java.util.List;

import com.edu.usquat.LookLearn.env.ImageUtils;
import com.edu.usquat.LookLearn.env.Logger;
import com.edu.usquat.LookLearn.tflite.Classifier;
import com.edu.usquat.LookLearn.tflite.TFLiteObjectDetectionEfficientDet;
import com.edu.usquat.R;

import androidx.core.content.res.ResourcesCompat;




public class LookLearnProcessor{

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
    private static final int TF_OD_API_INPUT_SIZE = 512;    //this is the wxh of square input size to MODEL
    private static final boolean TF_OD_API_IS_QUANTIZED = true;  //if its quantized or not. MUST be whatever the save tflite model is saved as

    //TFlite file for BodyPartCNN
    private static final String TF_OD_API_MODEL_FILE = "BodyPartCNN.tflite"; //"IRdetect.tflite";   //name of input file for MODEL must be tflite format
    //LabelMap file listed classes--same order as training
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/BodyPartCNNLabelMap.txt";



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
    public LookLearnProcessor(Context c ){
        this.context = c;

        //retrieve the percentages used in the forced attention Look Learn image creation
        backgroundPercent = ResourcesCompat.getFloat(context.getResources(), R.dimen.backgroundPercent);
        bodyPercent = ResourcesCompat.getFloat(context.getResources(), R.dimen.bodyPercent);
        bodyPartPercent = ResourcesCompat.getFloat(context.getResources(), R.dimen.bodyPartPercent);


        //setup transformation to crop images for processing

    }

    /**
     * Creates a forced attention LookLearn Image pased on the visualization percentages
     * represented by backgroundPercent,
     * input_frames = list of image bitmaps
     * @return list of forced attention Look Learn Images
     */
    public List<Bitmap> createAttentionImages(List<Bitmap> input_frames) {

        //saftey check
        if(input_frames.size() >0)
            return input_frames;

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

        //get the image size from the first image in the input_frames
        int inputWidth = input_frames.get(0).getWidth();
        int inputHeight = input_frames.get(0).getHeight();

        ++timestamp;
        final long currTimestamp = timestamp;
        Bitmap output;
        int pixel;  //stored rgb as a combined value using Color.argb format
        int A,R,G,B;


        //Now need to cycle through the frames in input_frames and run through the BodyPartCNN
        //first we need to resize it and then run it through the bodypartCNN
        for(int i=0; i<input_frames.size(); i++){

            //STEP 1: resize to run through body part CNN
            Bitmap b = Bitmap.createScaledBitmap(input_frames.get(0),cropSize,cropSize,true);
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

            //STEP3: create the forced Attension image stored in b that uses the detections
            // SEE PYTHON CODE -- ANKUSH we did not do ANY and ALL recognitions above the threshold --only one body, etc.
            //cycling through all of the recognition detections in my image I am currently processing
            for (final Classifier.Recognition result : results) {  //loop variable is result, represents one detection

                final RectF location = result.getLocation();  //getting as  a rectangle the bounding box of the result detecgiton

                //IF the detection is a body and not been used
                if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {

                    String title = result.getTitle();
                    if(title.contains("bodySquat") || title.contains("bodyTall")){

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
                        for (int x = left; x < right; ++x) {
                            for (int y =top; y < bottom; ++y) {
                                // get pixel color from original image
                                pixel = b.getPixel(x, y);
                                // apply filtering on each channel R, G, B
                                A = Color.alpha(pixel);
                                R = (int) (Color.red(pixel) * bodyPercent);
                                G = (int) (Color.green(pixel) * bodyPercent);
                                B = (int) (Color.blue(pixel) * bodyPercent);
                                // set new color pixel to output bitmap
                                output.setPixel(x, y, Color.argb(A, R, G, B));
                            }
                        }


                    }
                    //repeat for body parts
                }



            //set the LookLearn image to replace the original
            input_frames.set(i, output);

        }//end process this frames classification results



    }//end processing all of the frames

    return input_frames;
  }//end of create** method


}

