package com.edu.usquat.LookLearn.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import com.edu.usquat.LookLearn.env.Logger;

/**
 * Modified wrapper to accomodate for EfficientDet's output tensor, (tested on D0-512x512)
 * See: https://tfhub.dev/tensorflow/efficientdet/d0/1 for documentation on EfficientDet and its
 * output tensor information
 *
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * - https://github.com/tensorflow/models/tree/master/research/object_detection
 * where you can find the training code.
 *
 * To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
public class TFLiteObjectDetectionEfficientDet implements Classifier {

    private static final String TAG = "TFLiteObjectDetectionED";

    private static final Logger LOGGER = new Logger();

    // Only return this many results. THIS HAS TO MATCH THE MAX DETECTIONS PARAM FROM THE CONFIG FILE
    private static final int NUM_DETECTIONS = 100;
    // Float model
    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;
    // Number of threads in the java app
    private static final int NUM_THREADS = 4;
    private boolean isModelQuantized;
    // Config values.
    private int inputSize;
    public static int NUM_CLASSES = -1; // number of classes determined by reading label map file

    // THIS PARAMETER WAS DETERMINED BY LOOKING AT LOGS FROM OUTPUT TENSORS IN RecognizeImage() below
    // It will say something like:
    //  Cannot copy from a TensorFlowLite tensor (StatefulPartitionedCall:7) with shape [1, NUM_RAW_DETECTION_BOXES, 1] to a Java object with shape [x,x,x,x,x,x]
    private int NUM_RAW_DETECTION_BOXES = 49104; // see https://tfhub.dev/tensorflow/efficientdet/d0/1
    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;


    /*
    FOR THE FOLLOWING OUTPUT TENSORS:
        M : NUMBER OF RAW DETECTIONS (ARCHITECTURE-DEPENDENT)
        N : NUMBER OF DETECTIONS FROM CONFIG FILE (TRAINING-DEPENDENT)
        C : NUMBER OF CLASSES FROM CONFIG FILE (TRAINING-DEPENDENT)
     */
    // float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
    private float[][][] outputLocations;
    // int tensor of shape [N] containing detection class index from the label file.
    private float[][] outputClasses;
    // float32 tensor of shape [N] containing detection scores.
    private float[][] outputScores;
    // int tensor with only one value, the number of detections [N].
    private float[] numDetections;
    // float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
    float[][][] rawDetectionBoxes;
    // float32 tensor of shape [1, M, C] and contains class score logits for raw detection boxes. M is the number of raw detections.
    float[][][] rawDetectionScores;
    // float32 tensor of shape [1, N, C] and contains class score distribution (including background) for detection boxes in the image including background class.
    float[][][] detectionMulticlassScores;
    // float32 tensor of shape [N] and contains the anchor indices of the detections after NMS.
    float[][] detectionAnchorIndeces;

    private ByteBuffer imgData;

    private Interpreter tfLite;

    private TFLiteObjectDetectionEfficientDet() {}

    /** Memory-map the model file in Assets. */
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputSize The size of image input
     * @param isQuantized Boolean representing model is quantized or not
     */
    public static Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final String labelFilename,
            final int inputSize,
            final boolean isQuantized)
            throws IOException {
        final TFLiteObjectDetectionEfficientDet d = new TFLiteObjectDetectionEfficientDet();

        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(actualFilename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;

        // count number of lines in label map file
        int classNum = 0;

        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            d.labels.add(line);
            classNum ++;
        }
        br.close();

        TFLiteObjectDetectionEfficientDet.NUM_CLASSES = classNum;

        d.inputSize = inputSize;

        try {
            d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
            Log.d("MANNY", modelFilename);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.isModelQuantized = isQuantized;
        // Pre-allocate buffers.
        int numBytesPerChannel;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        // ONLY HAVE ONE COLOR CHANNEL
        d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
        d.imgData.order(ByteOrder.nativeOrder());
        d.intValues = new int[d.inputSize * d.inputSize];

        d.tfLite.setNumThreads(NUM_THREADS);
        d.outputLocations = new float[1][NUM_DETECTIONS][4];
        d.outputClasses = new float[1][NUM_DETECTIONS];
        d.outputScores = new float[1][NUM_DETECTIONS];
        d.numDetections = new float[1];
        return d;
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        try {
            imgData.rewind();
            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < inputSize; ++j) {
                    int pixelValue = intValues[i * inputSize + j];
                    if (isModelQuantized) {
                        // Quantized model
                        imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                        imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                        imgData.put((byte) (pixelValue & 0xFF));
                    } else { // Float model
                        imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                        imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                        imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    }
                }
            }
        }catch(Exception e) {
            System.out.println(e.toString());
        }
        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        outputLocations = new float[1][NUM_DETECTIONS][4];
        detectionMulticlassScores = new float[1][NUM_DETECTIONS][NUM_CLASSES];
        outputScores = new float[1][NUM_DETECTIONS];
        outputClasses = new float[1][NUM_DETECTIONS];
        detectionAnchorIndeces = new float[1][NUM_DETECTIONS];
        rawDetectionScores = new float[1][NUM_RAW_DETECTION_BOXES][NUM_CLASSES];
        numDetections = new float[1];
        rawDetectionBoxes = new float[1][NUM_RAW_DETECTION_BOXES][4];

        Object[] inputArray = {imgData};
        Map<Integer, Object> outputMap = new HashMap<>();

        outputMap.put(0, outputScores);         // [1,N] - Confidence Values
        outputMap.put(1, rawDetectionBoxes);    // [1,M,4] - Non-max-suppressed bounding boxes
        outputMap.put(2, numDetections);        // [N] - Number of detections
        outputMap.put(3, outputLocations);      // [1,N,4] - Bounding Boxes
        outputMap.put(4, outputClasses);        // [1,N] - class index
        outputMap.put(5, rawDetectionScores);   // [1,M,C] - raw detection scores
        outputMap.put(6, detectionMulticlassScores); // [1,N,C] - detection multiclass scores
        outputMap.put(7, detectionAnchorIndeces); // [1,N] detection anchor indeces

        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
        Trace.endSection();

        // Show the best detections.
        // after scaling them back to the input size.

        // You need to use the number of detections from the output and not the NUM_DETECTONS variable declared on top
        // because on some models, they don't always output the same total number of detections
        // For example, your model's NUM_DETECTIONS = 20, but sometimes it only outputs 16 predictions
        // If you don't use the output's numDetections, you'll get nonsensical data
        int numDetectionsOutput = Math.min(NUM_DETECTIONS, (int) numDetections[0]); // cast from float to integer, use min for safety

        final ArrayList<Recognition> recognitions = new ArrayList<>(numDetectionsOutput);

        // SSD Mobilenet V1 Model assumes class 0 is background class
        // in label file and class labels start from 1 to number_of_classes+1,
        // while outputClasses correspond to class index from 0 to number_of_classes
        // final int LABEL_OFFSET = -1;

        // log new frame process
        Log.d(TAG, "~~~~~~~~~~~~~~~~~ NEW FRAME ~~~~~~~~~~~~~~~~~~~~");

        final int LABEL_OFFSET = -1;

        for (int i = 0; i < numDetectionsOutput; ++i) {
            float score = outputScores[0][i];

            // log a positive result
            if( i < 3 && score > .5) {
                Log.d(TAG, "SCORE: " + String.valueOf(outputScores[0][i]));
            }

            final RectF detection =
                    new RectF(
                            outputLocations[0][i][1] * inputSize,
                            outputLocations[0][i][0] * inputSize,
                            outputLocations[0][i][3] * inputSize,
                            outputLocations[0][i][2] * inputSize);

            recognitions.add(new Recognition(
                    "" + i,
                    labels.get((int) outputClasses[0][i] + LABEL_OFFSET),
                    outputScores[0][i],
                    detection));
        }

        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }

    @Override
    public void enableStatLogging(final boolean logStats) {}

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {}

    public void setNumThreads(int num_threads) {
        if (tfLite != null) tfLite.setNumThreads(num_threads);
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {
        // current version of tfnightly does not recognize this method
        //if (tfLite != null) tfLite.setUseNNAPI(isChecked);
    }
}
