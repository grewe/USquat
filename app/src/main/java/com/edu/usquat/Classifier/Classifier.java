//package com.edu.usquat.Classifier;
//
//import android.app.Activity;
//import android.content.res.AssetFileDescriptor;
//import android.content.res.AssetManager;
//import android.os.FileUtils;
//
//import java.io.File;
//import java.io.FileInputStream;
//import java.io.IOException;
//import java.nio.MappedByteBuffer;
//import java.nio.channels.FileChannel;
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Map;
//import java.util.logging.Logger;
//
//import org.tensorflow.lite.DataType;
//import org.tensorflow.lite.Interpreter;
//import org.tensorflow.lite.gpu.CompatibilityList;
//import org.tensorflow.lite.gpu.GpuDelegate;
//import org.tensorflow.lite.support.common.FileUtil;
//
//
///*
//* This is the abstract class for handling inference and prediction
//* */
//public abstract class Classifier {
//    // TODO: implement Logger helper function
//    private static final Logger LOGGER = new Logger();
//
//    /* Enum type of devices */
//    public enum Device{
//        CPU,
//        GPU
//    }
//
//
//    /* Number of display result */
//    // TODO: have to recheck
//    private static final int MAX_RESULT = 3;
//
//    /* Loaded tflite Models */
//    private MappedByteBuffer feature_extractor_model;
//    private MappedByteBuffer lstm_model;
//
//    /* Images coordinator along x and y axis */
//    private final int image_size_x;
//    private final int getImage_size_y;
//
//    /* Input Size of images for feature_extractor_model (EfficientNet-B4) */
//    final int input_size = 380;
//
//    private final int num_frames;
//    private final int feature_vector_length;
//
//
//    /* Initialize an instance to run inference with TensorFlow Lite */
//    protected Interpreter feature_extractor_tflite;
//    protected Interpreter lstm_tflite;
//
//    /* Optional GPU Delegate for acceleration */
//    // Initialize interpreter with GPU delegate
//    // TODO: need to re-implement this
//    Interpreter.Options options = new Interpreter.Options();
//    GpuDelegate gpuDelegate;
//
//
//    final AssetManager assetManager = null;
//    final float confidence = 0.5f;
//    final int classes;
//
//    /* Labels corresponding to the probabilities output of LSTM model*/
//    private List<String> labels;
//
//    /* Initialize input of LSTM model input ( must be 3D: samples, time_steps, features)*/
//    public float[][][] lstm_input = null;
//
//    private static final float IMAGE_MEAN = 128.0f;
//    private static final float IMAGE_STD = 128.0f;
//
//
//    /* Memory Mapp the model file in Assets */
//    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFileName) throws IOException{
//        AssetFileDescriptor fileDescriptor = assets.openFd(modelFileName);
//        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
//        FileChannel fileChannel = inputStream.getChannel();
//        long startOffset = fileDescriptor.getStartOffset();
//        long declareLength = fileDescriptor.getDeclaredLength();
//        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declareLength);
//    }
//
//    /*Create a Classifer with specific configuration */
//
//    public static Classifier create(Activity activity,Device device,int numThreads,AssetManager assetManager) throws IOException{
//        return new ClassifierFloat(activity,device,numThreads,assetManager);
//    }
//
//    /*Create an immutable results obtained from Classifier */
//    public static class Recognition{
//        private final String id;
//        private final String title;
//        private final float confidence;
//
//        public Recognition(final String id, final String title, final Float confidence){
//            this.id = id;
//            this.title = title;
//            this.confidence = confidence;
//        }
//
//        /*Get methods for class Recognition*/
//        public String getID(){
//            return id;
//        }
//
//        public String getTitle(){
//            return title;
//        }
//
//        public float getConfidence(){
//            return confidence;
//        }
//
//        @Override
//        public String toString(){
//            String resultString = "";
//
//            if (id!=null){
//                resultString += "[" + id + "]";
//            }
//            if (title!=null){
//                resultString += title + " ";
//            }
//            if (confidence!=0){
//                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
//            }
//            return resultString.trim();
//        }
//
//    }
//
//    /* Feature Extractor Model is loading here */
//
//    protected Classifier (Activity activity,Device device,int numThreads,AssetManager assetManager) throws IOException{
//        //TODO: need to change feature_extractor.tflite path
//        feature_extractor_model = FileUtil.loadMappedFile(activity,'feature_extractor.tflite');
//        switch (device){
//            case GPU:
//                gpuDelegate = new GpuDelegate();
//                options.addDelegate(gpuDelegate);
//                break;
//            case CPU:
//                break;
//        }
//        options.setNumThreads(numThreads);
//
//
//        // Create Intepreter Instance
//        feature_extractor_tflite = new Interpreter(feature_extractor_model,options);
//        //TODO
//        /* Extract features given the video here*/
//
//        int imageTensorIndex = 0;
//        // TODO: need to re-check shape & type here
//        int[] imageShape = feature_extractor_tflite.getInputTensor(imageTensorIndex).shape();
//        DataType imageDataType = feature_extractor_tflite.getInputTensor(imageTensorIndex).dataType();
//
//
//        /*Output of feature_extractor_tflite (feature_Vector)*/
//        int feature_vector_index = 0;
//        // TODO: check -- have to match with {1,1792}
//        int[] feature_vector_shape = feature_extractor_tflite.getOutputTensor(feature_vector_index).shape();
//        DataType feature_vector_type = feature_extractor_tflite.getOutputTensor(feature_vector_index).dataType();
//
//
//        /*LSTM models is loaded here*/
//        // TODO: Change filepath of LSTM.tflite
//        lstm_model = FileUtil.loadMappedFile(activity,"lstm.tflite");
//
//        // Create Intepreter Instance
//        lstm_tflite = new Interpreter(lstm_model,options);
//
//        // Load class labels from file
//        // TODO: need to add class_labels path
//        labels = FileUtil.loadLabels(activity,"class_labels.txt");
//        classes = labels.size();
//
//        int imageTensorIndex2 = 0;
//        // TODO: need to check shape {40,16,width,3}
//        int[] imageShape2 = lstm_tflite.getInputTensor(imageTensorIndex2).shape();
//
//        // TODO: need to check
//        num_frames = imageShape2[1];
//        feature_vector_length = imageShape[2];
//
//        // Define LSTM input here [frames][16][feature_vector_length]
//        lstm_input = new float[numThreads][16][feature_vector_length];
//    }
//
//    //TODO: here LSTM has to take a processed_image_buffer getting from the recorded video
//
//    /*Here we start calling the final classification result*/
//    public List<Recognition> lstmClassifer(){
//
//        // Passing image_buffer to lstm tflite.run(input,output)
//        lstm_tflite.run(images_buffer,prob_list);
//        // Map labels with prob_list
//        Map<String,Float> probs = makeProb(prob_list[0]);
//        return getTopKProbability(probs);
//    }
//
//    public Map<String,Float> makeProb(float[] list){
//        return result;
//    }
//
//    static List<Recognition> getTopKProbability(Map<String,Float> prob_list){
//        // find the top max probabilities using max heap
//        final ArrayList<Recognition> recognitionArrayList = new ArrayList<>();
//        // TODO: implement max heap here
//        return recognitionArrayList;
//    }
//
//
//}
//
