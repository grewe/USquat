package com.edu.usquat.Classifier;
import android.animation.TimeInterpolator;
import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.FileUtils;
import android.util.Log;
import android.util.TimingLogger;

import com.edu.usquat.R;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/** A classifier specialized to label images using TensorFlow Lite.
 *
 *    NOTE 2: Classifier contains most of the complex logic for processing the camera input and running inference.
 *
 *       A subclasses of the file exist, in ClassifierFloatMobileNet.java (in other Tensorflowlite examples there is ClassifierQuantizedMobileNet.java), to demonstrate the use of
 *       floating point (and quantized) models.
 *
 *       The Classifier class implements a static method, create, which is used to instantiate the appropriate subclass based on the supplied model type (quantized vs floating point).
 *
 *
 */

public abstract class Classifier {

    private static final String TAG = "Classififer" ;

    /**
     * The runtime device type used for executing classification.
     */
    public enum Device {
        CPU,
        GPU
    }

    /** Number of results to show in the UI. */
    private static final int MAX_RESULTS = 2;

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer featureExtractorModel;
    private MappedByteBuffer lstmModel;
    private final int featureLength;
    private GpuDelegate gpuDelegate = null;
    final int inputSize = 380;

    private final int frames;
    /** Input image TensorBuffer. */
    private TensorImage inputImageBuffer;

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter extractorTflite;
    protected Interpreter lstmTflite;
    final AssetManager assetManager = null;
    final float confidenceInterval = 0.5f;


    //parameter to select the LSTM Model either Original (kelly) or LookLearn(ankush)
    private String mUSquatModel;

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /** Labels corresponding to the output of the vision model. */
    private List<String> labels;

    public float[][][] lstmInput = null;


    public int frame = 0; // FRAME NUMBER
    public int frame2 = 0;


    /** Load Tflite Model */
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
     * Creates a classifier with the provided configuration.
     *
     * @param activity The current Activity.
     * @param device The device to use for classification.
     * @param numThreads The number of threads
     *
     * @return A classifier with the desired configuration.
     */
    public static Classifier create(Activity activity, Device device, int numThreads, AssetManager assetManager)
            throws IOException {

       return new ClassifierFloat(activity, device, numThreads, assetManager);

    }

    /** An immutable result returned by a Classifier describing what was recognized. */
    public static class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /** Display name for the recognition. */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;


        public Recognition(
                final String id, final String title, final Float confidence) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;

        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }


        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }


            return resultString.trim();
        }
    }
    /** Initializes a {@code Classifier}.
     *
     *
     * To perform inference, we need to load a model file and instantiate an Interpreter.
     * This happens in the constructor of the Classifier class, along with loading the list of class labels.
     * Information about the device type and number of threads is used to configure the Interpreter via the
     * Interpreter.Options instance passed into its constructor. Note how that in the case of a GPU being
     * available, a Delegate is created using GpuDelegateHelper.
     *
     * */
    protected Classifier(Activity activity, Device device, int numThreads,AssetManager assetManager) throws IOException {

        //read in the USquatModel specification from the strings folder (strings.xml)  of the application
        this.mUSquatModel= activity.getApplicationContext().getResources().getString(R.string.USquatModel);
        Log.d(TAG,"USquatModel option: "+this.mUSquatModel);


        featureExtractorModel = FileUtil.loadMappedFile(activity, getModelPath());
        switch (device) {
            case GPU:
                //create a GPU delegate instance and add it to the interpreter options
                gpuDelegate = new GpuDelegate();
                tfliteOptions.addDelegate(gpuDelegate);

                break;
            case CPU:
                break;
        }
        tfliteOptions.setNumThreads(numThreads);

        // Create a TFLite interpreter instance
      extractorTflite = new Interpreter(featureExtractorModel, tfliteOptions);

        int imageTensorIndex = 0;
        int[] imageShape = extractorTflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3} get input tensor

        DataType imageDataType = extractorTflite.getInputTensor(imageTensorIndex).dataType();


        int probabilityTensorIndex = 0;
        int[] probabilityShape =
                extractorTflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, 1792} the shape of output
        DataType probabilityDataType = extractorTflite.getOutputTensor(probabilityTensorIndex).dataType(); // datatype

        // Creates the input tensor.
        inputImageBuffer = new TensorImage(imageDataType);

// **********************************CREATE LSTM MODEL HERE ***************************************

        if(this.mUSquatModel == "Original")
            lstmModel = FileUtil.loadMappedFile(activity, "lstm_classifier.tflite");
        else
            lstmModel = FileUtil.loadMappedFile(activity, "lstm_lookLearn_classifier.tflite");


        // Create a TFLite interpreter instance
        lstmTflite = new Interpreter(lstmModel, tfliteOptions);

        // Loads labels out from the label file.
        labels = FileUtil.loadLabels(activity, getLabelPath()); // Igonore Possibly?????
        //classes = labels.size();

        int imageTensorIndex2 = 0;
        int[] imageShape2 = lstmTflite.getInputTensor(imageTensorIndex2).shape(); // must look smt like this {1,40,1792} get input tensor

        frames = imageShape2[1]; // number of frames
        featureLength = imageShape2[2]; //
        lstmInput = new float[frames][1][featureLength];

        // TODO
    }
        private List<Recognition> test;
        public List<Recognition> getFramesAndProcess(final List<Bitmap> processing_frames ){
        Log.d(TAG,String.valueOf(processing_frames.size()));
            return recognizeImages(processing_frames);

        }

        // TODO: sliding window for
        /** Runs inference and returns the classification results.
         * */
        public List<Recognition> recognizeImages(final List<Bitmap> processing_frames) {
//
//            int max_frame = 40;
//            int whole = processing_frames.size() / 40;
//            double fraction = (double) Math.round(((processing_frames.size() / 40.0) % 1) * 100)/100 ;
//            boolean odd = true;
//            int current_frame = 1;
//            int sample_every_k_frame = Math.max(1, whole);
//            int step = 0;
//            while (true) {

//                if (fraction >= 0.0 && fraction <= 0.3 && current_frame < processing_frames.size() - whole) {
//                    if (current_frame % sample_every_k_frame == 0) {
//                        current_frame += sample_every_k_frame;
//                        inputImageBuffer = loadImage(processing_frames.get(current_frame));
//                        extractorTflite.run(inputImageBuffer.getBuffer(), lstmInput[step]);
//                        if(current_frame == processing_frames.size() || current_frame > processing_frames.size()){
//                            break;
//                        }
//                        step += 1;
//                        max_frame -= 1;
//                    }
//                } else if (fraction <= 0.7 && fraction > 0.3 && current_frame < processing_frames.size() - sample_every_k_frame + 1) {
//                    if (odd) {
//                        current_frame += sample_every_k_frame;
//                        odd = false;
//                        inputImageBuffer = loadImage(processing_frames.get(current_frame));
//                        extractorTflite.run(inputImageBuffer.getBuffer(), lstmInput[step]);
//                        if(current_frame == processing_frames.size() || current_frame > processing_frames.size()){
//                            break;
//                        }
//                        step += 1;
//                        max_frame -= 1;
//
//                    } else {
//                        current_frame += (whole + 1);
//                        odd = true;
//                        inputImageBuffer = loadImage(processing_frames.get(current_frame));
//                        extractorTflite.run(inputImageBuffer.getBuffer(), lstmInput[step]);
//                        if(current_frame == processing_frames.size()|| current_frame > processing_frames.size()){
//                            break;
//                        }
//                        step += 1;
//                        max_frame -= 1;
//
//                    }
//                } else if (fraction < 0.7 && current_frame < processing_frames.size() - (whole + 3)) {
//                    if (odd) {
//                        current_frame += whole;
//                        odd = false;
//                        inputImageBuffer = loadImage(processing_frames.get(current_frame));
//                        extractorTflite.run(inputImageBuffer.getBuffer(), lstmInput[step]);
//                        if (current_frame == processing_frames.size()|| current_frame > processing_frames.size()) {
//                            break;
//                        }
//                        step += 1;
//                        max_frame -= 1;
//
//                    } else {
//                        current_frame += (whole + 2);
//                        odd = true;
//                        inputImageBuffer = loadImage(processing_frames.get(current_frame));
//                        extractorTflite.run(inputImageBuffer.getBuffer(), lstmInput[step]);
//                        if (current_frame == processing_frames.size()|| current_frame > processing_frames.size()) {
//                            break;
//                        }
//                        step += 1;
//                        max_frame -= 1;
//                    }
//                }
////                } else{
////                    break;
////                }
//
//                if (max_frame == 1 || step == 39 ) {
//                    break;
//                }
//
//            }
//            return recognizeImageLSTM();
//        }

            // Only get the first 40 frames.
             Log.d(TAG,String.valueOf(String.format("%1$TH:%1$TM:%1$TS",System.currentTimeMillis())));
            for (int i = 0; i < 40; i++) {
                if (i == processing_frames.size()) {
                    Log.d(TAG, String.valueOf(String.format("%1$TH:%1$TM:%1$TS", System.currentTimeMillis())));
                    float ratio = processing_frames.size() / 40;
                    Log.d(TAG, String.valueOf(ratio));
                    return recognizeImageLSTM();
                }
                inputImageBuffer = loadImage(processing_frames.get(i));
                extractorTflite.run(inputImageBuffer.getBuffer(), lstmInput[i]);

            }
            Log.d(TAG, String.valueOf(String.format(" start LSTM: %1$TH:%1$TM:%1$TS", System.currentTimeMillis())));
            return recognizeImageLSTM();
        }







//         return null;
//        }
    /** Closes the interpreter and model to release resources. */
    public void close() {
        if (extractorTflite != null) {
            // Close the interpreter
            extractorTflite.close();
            extractorTflite = null;


        }
        if (lstmTflite != null) {
            lstmTflite.close();
            lstmTflite = null;

        }
        // Close the GPU delegate
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }


        featureExtractorModel = null;
        lstmModel = null;
    }
        private TensorImage loadImage(final Bitmap bitmap){
            inputImageBuffer.load(bitmap);
            ImageProcessor imageProcessor = new ImageProcessor.Builder()
                    .add(getPreprocessNormalizeOp())
                    .build();
            return imageProcessor.process(inputImageBuffer);
        }

        public List<Recognition> recognizeImageLSTM() {
            Log.d(TAG,String.valueOf(String.format("%1$TH:%1$TM:%1$TS",System.currentTimeMillis())));
            float[][] test = new float[1][5];
            // Run TFLite inference passing in the processed image.
            lstmTflite.run(lstmInput, test);

            Map<String, Float> labeledProbability = makeProb(test[0]);

            return getTopKProbability(labeledProbability);

        }


        public Map<String, Float> makeProb(float[] results) {
            Map<String,Float> temp = new HashMap<String, Float>();
            // get from labels.txt
            for (int i = 0; i < 5; i++) {
                temp.put(labels.get(i), results[i]);
            }
            return temp;
        }

    /** Gets the top-k results. */
    static List<Recognition> getTopKProbability(Map<String, Float> labelProb) {
        // Find the best classifications.
        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
            pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue()));
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i <recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }

    /** Gets the name of the model file stored in Assets. */
    protected abstract String getModelPath();

    /** Gets the name of the label file stored in Assets. */
    protected abstract String getLabelPath();

    /** Gets the TensorOperator to nomalize the input image in preprocessing. */
    protected abstract TensorOperator getPreprocessNormalizeOp();

    /**
     * Gets the TensorOperator to dequantize the output probability in post processing.
     *
     * <p>For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
     * essentially linear transformation). For float model, de-quantize is not required. But to
     * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
     * 1.0f, respectively.
     */
    protected abstract TensorOperator getPostprocessNormalizeOp();
    }