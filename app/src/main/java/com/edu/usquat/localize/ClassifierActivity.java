package com.edu.usquat.localize;

/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Button;
import android.widget.TextView;

import java.io.IOException;
import java.util.List;

import com.edu.usquat.R;
import com.edu.usquat.localize.env.BorderedText;
import com.edu.usquat.localize.env.Logger;
import com.edu.usquat.localize.tflite.Classifier;
import com.edu.usquat.localize.tflite.Classifier.Device;

public class ClassifierActivity extends com.edu.usquat.localize.CameraActivity implements OnImageAvailableListener {

    public static String imageFileURL;
    private static int op= 0;
    //    Queue<float[][]> qe = new LinkedList<float[][]>();
    // flag for pushing records
    int flag = 0;
    private static final Logger LOGGER = new Logger();
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final float TEXT_SIZE_DIP = 10;
    private Bitmap rgbFrameBitmap = null;
    Bitmap resized = null;
    static long lastProcessingTimeMs = 0;
    static long startTime = 0;
    private Integer sensorOrientation;
    private Classifier classifier;
    public static int counter = 0;
    public int frame = 0;
    public List<Classifier.Recognition> temp;
    private BorderedText borderedText;
    /** Input image size of the model along x axis. */
    private int imageSizeX;
    /** Input image size of the model along y axis. */
    private int imageSizeY;
    TextView ourText;
    Button emotionDetection;


    @Override
    protected int getLayoutId() {
        return R.layout.tfe_ic_camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        recreateClassifier(getDevice(), getNumThreads());
        if (classifier == null) {
            LOGGER.e("No classifier on preview!");
            return;
        }

        //previewWidth = size.getWidth();
        // previewHeight = size.getHeight();
        previewWidth = size.getWidth();
        previewHeight = size.getHeight();
        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    }

    /**
     * processImage = method that will process each Image
     *
     * NOTE: This ClassifierActivity extends CameraActivity that  gets the camera input using the functions defined in the file CameraActivity.java.
     * This file depends on AndroidManifest.xml to set the camera orientation.
     *
     * CameraActivity also contains code to capture user preferences from the UI and make them available to other classes via convenience methods.
     *
     * model = Model.valueOf(modelSpinner.getSelectedItem().toString().toUpperCase());
     * device = Device.valueOf(deviceSpinner.getSelectedItem().toString());
     * numThreads = Integer.parseInt(threadsTextView.getText().toString().trim());
     *
     * NOTE 2: The file Classifier.java contains most of the complex logic for processing the camera input and running inference.
     *      *
     *      * A subclasses of the file exist, in ClassifierFloatMobileNet.java (in other Tensorflowlite examples there is ClassifierQuantizedMobileNet.java), to demonstrate the use of
     *      * floating point (and quantized) models.
     *      *
     *      * The Classifier class implements a static method, create, which is used to instantiate the appropriate subclass based on the supplied model type (quantized vs floating point).
     *      *
     */

    @Override
    protected void processImage() {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final int cropSize = Math.min(previewWidth, previewHeight);
        flag++;

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        if (classifier != null) {
                            startTime = SystemClock.elapsedRealtime();
                            //SystemClock.uptimeMillis();
                            if (counter > 0)
                                counter += 1;


                            // TODO: need to modify the getFrames()


                            if (counter <= 12 && counter >= 2) {
                                temp = classifier.getFrames(rgbFrameBitmap, sensorOrientation);
                            }
                            if (counter == 50)
                                counter = 0;

                            runOnUiThread(
                                    new Runnable() {
                                        @Override
                                        public void run() {
                                            //DISPLAY information including recognition results
                                            if (temp != null)
                                                showResultsInBottomSheet(temp);

                                            showFrameInfo(previewWidth + "x" + previewHeight);
                                            showCropInfo(imageSizeX + "x" + imageSizeY);
                                            showCameraResolution(cropSize + "x" + cropSize);
                                            showRotationInfo(String.valueOf(sensorOrientation));
                                            showInference(counter + "ms");


                                        }
                                    });
                            lastProcessingTimeMs = startTime;
                        }
                        readyForNextImage();
                    }
                });
    }

    @Override
    protected void onInferenceConfigurationChanged() {
        if (rgbFrameBitmap == null) {
            // Defer creation until we're getting camera frames.
            return;
        }
        final Device device = getDevice();
        final int numThreads = getNumThreads();
        runInBackground(() -> recreateClassifier(device, numThreads));

    }

    private void recreateClassifier(Device device, int numThreads) {
        if (classifier != null) {
            LOGGER.d("Closing classifier.");
            classifier.close();
            classifier = null;
        }
        try {
            LOGGER.d(
                    "Creating classifier (device=%s, numThreads=%d)", device, numThreads);
            classifier = Classifier.create(this, device, numThreads,getAssets());
//            classifier2 = Classifier2.create(this, device, numThreads);
        } catch (IOException e) {
            LOGGER.e(e, "Failed to create classifier.");
        }

        // Updates the input image size.
        imageSizeX = classifier.getImageSizeX();
        imageSizeY = classifier.getImageSizeY();
    }
}