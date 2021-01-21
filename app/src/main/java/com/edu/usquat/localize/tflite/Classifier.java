/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.edu.usquat.localize.tflite;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.RectF;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;

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
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

import com.edu.usquat.localize.env.Logger;

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
  private static final Logger LOGGER = new Logger();

  /** The runtime device type used for executing classification. */
  public enum Device {
    CPU,
    GPU
  }

  /** Optional GPU delegate for acceleration. */
  private GpuDelegate gpuDelegate = null;

  /** Number of results to show in the UI. */
  private static final int MAX_RESULTS = 2;

  /**Float Model*/
  private static final float IMAGE_MEAN = 128.0f;
  private static final float IMAGE_STD = 128.0f;

  private final float confidenceInterval = 0.5f;
  private final int classes;


  /** The loaded TensorFlow Lite model getting from subclass ClassiferFloatMobileNet
   * -----included the FeatureExtractor + its label.
   * */
  private MappedByteBuffer featureExtractorModel;
  private MappedByteBuffer lstmModel;

  /** Image size along the x axis. */
  private final int imageSizeX;
  private final int frames;


  /** Image size along the y axis. */
  private final int imageSizeY;
  private final int featureLength;



  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter featureTflite;
  protected Interpreter faceTflite;
  protected Interpreter lstmTflite;
  final AssetManager assetManager = null;


  /** Options for configuring the Interpreter. */
  private final Interpreter.Options tfliteOptions = new Interpreter.Options();

  /** Labels corresponding to the output of the vision model. */
  private List<String> labels;

  /** Input image TensorBuffer. */
  private TensorImage inputImageBuffer;

  public float[][][] lstmInput = null; // LSTM INPUT VERY IMPORTANT
  public int frame = 0; // FRAME NUMBER
  public int frame2 = 0;

  /** Define inputs for EfficientDet model (human/face detection)
   * */
  public static int NUM_DETECTIONS = 10; // Must match the train model
  public final int inputSize = 512; // input size for EfficientDet Model (512x512)

  float[][][] outputLocations = new float[1][NUM_DETECTIONS][4];
  float[][] outputClasses = new float[1][NUM_DETECTIONS];
  float[][] outputScores = new float[1][NUM_DETECTIONS];
  float[] numDetections = new float[1];


  // TODO: need to modify along with getframe() method
  Bitmap resizedbitmap = null;
  Bitmap image = null;
  Bitmap resized;
  Bitmap bitmap1 = null;
  Bitmap bitmap2 = null;
  Bitmap bitmap3 = null;
  Bitmap bitmap4 = null;
  Bitmap bitmap5 = null;
  Bitmap bitmap6 = null;
  Bitmap bitmap7 = null;
  Bitmap bitmap8 = null;
  Bitmap bitmap9 = null;
  Bitmap bitmap10 = null;
  private Vector<String> labels2 = new Vector<String>();
//  private Vector<String> labels = new Vector<String>();

  int[] sensorArray = new int[10];
  int foundFace = 0;

  private int[] intValues;
  //  public int inputSize = 512;
  private ByteBuffer imgData;


  /** Load TensorflowLite Model from assets and return ByteBuffer object
   * */
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
   * @param numThreads The number of threads to use for classification.
   * @return A classifier with the desired configuration.
   */
  public static Classifier create(Activity activity, Device device, int numThreads, AssetManager assetManager)
          throws IOException {

    return new ClassifierFloatMobileNet(activity, device, numThreads, assetManager);
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

    /** Optional location within the source image for the location of the recognized object. */
    private RectF location;

    public Recognition(
            final String id, final String title, final Float confidence, final RectF location) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
      this.location = location;
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

    public RectF getLocation() {
      return new RectF(location);
    }

    public void setLocation(RectF location) {
      this.location = location;
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

      if (location != null) {
        resultString += location + " ";
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
  protected Classifier(Activity activity, Device device, int numThreads, AssetManager assetManager) throws IOException {
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
    featureTflite = new Interpreter(featureExtractorModel, tfliteOptions);

    int imageTensorIndex = 0;
    int[] imageShape = featureTflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3} get input tensor


    imageSizeY = imageShape[1];
    imageSizeX = imageShape[2];

    DataType imageDataType = featureTflite.getInputTensor(imageTensorIndex).dataType();


    int probabilityTensorIndex = 0;
    int[] probabilityShape =
            featureTflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, 2048} the shape of output
    DataType probabilityDataType = featureTflite.getOutputTensor(probabilityTensorIndex).dataType(); // datatype

    // Creates the input tensor.
    inputImageBuffer = new TensorImage(imageDataType);

  /**
   * DEFINE LSTM MODEL (inputs & outputs)
   * */
  // TODO: modify based on trained LSTM Model

    // Load Model
    lstmModel = FileUtil.loadMappedFile(activity, "lstmModel.tflite");

    // Create a TFLite interpreter instance
    lstmTflite = new Interpreter(lstmModel, tfliteOptions);

    // Loads labels out from the label file.
    labels = FileUtil.loadLabels(activity, getLabelPath()); // Igonore Possibly?????
    classes = labels.size();

    int imageTensorIndex2 = 0;
    int[] imageShape2 = lstmTflite.getInputTensor(imageTensorIndex2).shape(); // {10, 1, width, 3} get input tensor

    frames = imageShape2[1];
    featureLength = imageShape2[2];
    lstmInput = new float[frames][1][featureLength];

/** DEFINE HUMAN/FACE MODEL (EfficientDet D0)
 * */

    String modelFilename = "efficientDet.tflite";
    String labelFilename = "file:///android_asset/labelmap.txt";


    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    InputStream labelsInput = assetManager.open(actualFilename);
    BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      labels2.add(line);
    }
    br.close();

    faceTflite = new Interpreter(loadModelFile(assetManager, modelFilename));

    int numBytesPerChannel = 4; // Floating point

    // Has to be ByteBuffer object
    imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * numBytesPerChannel);
    imgData.order(ByteOrder.nativeOrder());
    intValues = new int[inputSize * inputSize];

  }

  /** RUN INFERENCE LSTM MODEL & return classification results
   * */
  public List<Recognition> recognizeImageLSTM() {
    float[][] test = new float[1][classes];
    // Run TFLite inference passing in the processed image.
    lstmTflite.run(lstmInput, test);

    Map<String, Float> labeledProbability = makeProb(test[0]);

    return getTopKProbability(labeledProbability);
  }


  public Map<String, Float> makeProb(float[] results) {
    Map<String, Float> temp = new HashMap<String, Float>();
//    String[] theLabels = {"disgust", "fear", "happiness", "others", "repression", "sadness", "surprise"};
    for (int i = 0; i < classes; i++) {
      temp.put(labels.get(i) + " " + String.valueOf(foundFace), results[i]);
    }
    return temp;
  }

  /** RUN INFERENCE OF HUMAN/FACE MODEL
   * */
  public List<Recognition> recognizeFaceImage(final Bitmap bitmap, int sensorOrientation) {
    intValues = new int[inputSize * inputSize];
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];

        imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);

      }
    }

    // Copy the input data into Tensorflow
    outputLocations = new float[1][NUM_DETECTIONS][4];
    outputClasses = new float[1][NUM_DETECTIONS];
    outputScores = new float[1][NUM_DETECTIONS];
    numDetections = new float[1];

    Object[] inputArray = {imgData};
    Map<Integer, Object> outputMap = new HashMap<>();

    // Map input - output
    outputMap.put(0, outputLocations);
    outputMap.put(1, outputClasses);
    outputMap.put(2, outputScores);
    outputMap.put(3, numDetections);
    faceTflite.runForMultipleInputsOutputs(inputArray, outputMap);

    int numDetectionsOutput = Math.min(NUM_DETECTIONS, (int) numDetections[0]); // cast from float to integer, use min for safety
    int max = 0;
    if (numDetectionsOutput > 0) {
      max = 0;
      for (int i = 0; i < numDetectionsOutput; i++) {
        if (outputScores[0][max] < outputScores[0][i]) {
          max = i;
        }
      }
    }

    /**IF detect something with confident > 0.5f , feed into recognizeImage() which will run the Feature Extractor
     * */
    if (numDetectionsOutput > 0 && outputScores[0][max] > confidenceInterval) {
      int left = (int)(outputLocations[0][max][1] * inputSize);
      int top = (int)(outputLocations[0][max][0] * inputSize);
      int width = ((int)(outputLocations[0][max][3] * inputSize)) - left;
      int height = ((int)(outputLocations[0][max][2] * inputSize)) - top;
      if ((width + left) < inputSize && (height + top) < inputSize && left > -1 && top > -1) {
        resizedbitmap = Bitmap.createBitmap(bitmap, left, top, width, height);
        resizedbitmap = Bitmap.createScaledBitmap(resizedbitmap, imageSizeX, imageSizeY, true);
        foundFace += 1;

      } else {
        resizedbitmap = Bitmap.createScaledBitmap(bitmap, imageSizeX, imageSizeY, true);
      }

      return recognizeImage(resizedbitmap, sensorOrientation);
    } else {
      resizedbitmap = Bitmap.createScaledBitmap(bitmap, imageSizeX, imageSizeY, true);
      return recognizeImage(resizedbitmap, sensorOrientation);
    }
//    return recognitions;
  }

  /** Runs feature Extractor model inference and feed results to LSTM Model (defined in recognizeImageLSTM) */
  public List<Recognition> recognizeImage(final Bitmap bitmap, int sensorOrientation) {
    if (frame2 < frames) {
      inputImageBuffer = loadImage(bitmap, sensorOrientation);
      featureTflite.run(inputImageBuffer.getBuffer(), lstmInput[frame2]);

      if (frame2 == 9) {
        frame2 = 0;
        return recognizeImageLSTM();
      }
      frame2 += 1;
    }
    return null;

  }


  // TODO: need to redo
  public List<Recognition> getFrames(final Bitmap bitmap, int sensorOrientation) {
    if (frame == 0) {
      foundFace = 0;
      bitmap1 = Bitmap.createBitmap(bitmap);
      sensorArray[frame] = sensorOrientation;
    } else if (frame == 1) {
      bitmap2 = Bitmap.createBitmap(bitmap);
      sensorArray[frame] = sensorOrientation;
    } else if (frame == 2) {
      bitmap3 = Bitmap.createBitmap(bitmap);
      sensorArray[frame] = sensorOrientation;
    } else if (frame == 3) {
      bitmap4 = Bitmap.createBitmap(bitmap);
      sensorArray[frame] = sensorOrientation;
    } else if (frame == 4) {
      bitmap5 = Bitmap.createBitmap(bitmap);
      sensorArray[frame] = sensorOrientation;
    } else if (frame == 5) {
      bitmap6 = Bitmap.createBitmap(bitmap);
      sensorArray[frame] = sensorOrientation;
    } else if (frame == 6) {
      bitmap7 = Bitmap.createBitmap(bitmap);
      sensorArray[frame] = sensorOrientation;
    } else if (frame == 7) {
      bitmap8 = Bitmap.createBitmap(bitmap);
      sensorArray[frame] = sensorOrientation;
    } else if (frame == 8) {
      bitmap9 = Bitmap.createBitmap(bitmap);
      sensorArray[frame] = sensorOrientation;
    } else if (frame == 9) {
      bitmap10 = Bitmap.createBitmap(bitmap);
      sensorArray[frame] = sensorOrientation;
    } else {
      frame = 0;
      for (int i = 0; i < frames; i++) {
        if (i == 0) {
          bitmap1 = Bitmap.createScaledBitmap(bitmap1, inputSize, inputSize, true);
          bitmap1 = RotateBitmap(bitmap1,sensorArray[i]);
          recognizeFaceImage(bitmap1,sensorArray[i]);
        } else if (i == 1) {
          bitmap2 = Bitmap.createScaledBitmap(bitmap2,  inputSize, inputSize, true);
          bitmap2 = RotateBitmap(bitmap2,sensorArray[i]);
          recognizeFaceImage(bitmap2,sensorArray[i]);
        } else if (i == 2) {
          bitmap3 = Bitmap.createScaledBitmap(bitmap3, inputSize, inputSize, true);
          bitmap3 = RotateBitmap(bitmap3,sensorArray[i]);
          recognizeFaceImage(bitmap3,sensorArray[i]);
        } else if (i == 3) {
          bitmap4 = Bitmap.createScaledBitmap(bitmap4,  inputSize, inputSize, true);
          bitmap4 = RotateBitmap(bitmap4,sensorArray[i]);
          recognizeFaceImage(bitmap4,sensorArray[i]);
        } else if (i == 4) {
          bitmap5 = Bitmap.createScaledBitmap(bitmap5,  inputSize, inputSize, true);
          bitmap5 = RotateBitmap(bitmap5,sensorArray[i]);
          recognizeFaceImage(bitmap5,sensorArray[i]);
        } else if (i== 5) {
          bitmap6 = Bitmap.createScaledBitmap(bitmap6,  inputSize, inputSize, true);
          bitmap6 = RotateBitmap(bitmap6,sensorArray[i]);
          recognizeFaceImage(bitmap6,sensorArray[i]);
        } else if (i== 6) {
          bitmap7 = Bitmap.createScaledBitmap(bitmap7,  inputSize, inputSize, true);
          bitmap7 = RotateBitmap(bitmap7,sensorArray[i]);
          recognizeFaceImage(bitmap7,sensorArray[i]);
        } else if (i == 7) {
          bitmap8 = Bitmap.createScaledBitmap(bitmap8,  inputSize, inputSize, true);
          bitmap8 = RotateBitmap(bitmap8,sensorArray[i]);
          recognizeFaceImage(bitmap8,sensorArray[i]);
        } else if (i == 8) {
          bitmap9 = Bitmap.createScaledBitmap(bitmap9,  inputSize, inputSize, true);
          bitmap9 = RotateBitmap(bitmap9,sensorArray[i]);
          recognizeFaceImage(bitmap9,sensorArray[i]);
        } else if (i == 9) {
          bitmap10 = Bitmap.createScaledBitmap(bitmap10,  inputSize, inputSize, true);
          bitmap10 = RotateBitmap(bitmap10,sensorArray[i]);
          return recognizeFaceImage(bitmap10,sensorArray[i]);
        }
      }
    }

    frame+= 1;

    return null;
  }

  public static Bitmap RotateBitmap(Bitmap source, float angle)
  {
    Matrix matrix = new Matrix();
    matrix.postRotate(angle);
    return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
  }




  /** Closes the interpreter and model to release resources. */
  public void close() {
    if (featureTflite != null) {
      // Close the interpreter
      featureTflite.close();
      featureTflite = null;


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

  /** Get the image size along the x axis. */
  public int getImageSizeX() {
    return imageSizeX;
  }

  /** Get the image size along the y axis. */
  public int getImageSizeY() {
    return imageSizeY;
  }

  /** Loads input image, and applies preprocessing. */
  private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
    // Loads bitmap into a TensorImage.
    inputImageBuffer.load(bitmap);

    // Creates processor for the TensorImage.
    int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
    int numRoration = sensorOrientation / 90;
    // Define an ImageProcessor from TFLite Support Library to do preprocessing
    /* Basic image processor
    ImageProcessor imageProcessor =
            new ImageProcessor.Builder()




                .build();
    return imageProcessor.process(inputImageBuffer);*/

    //Image processor that resizes to cropSizeXcropSize or to imageSizeX X imageSizeY, and that can rotate 90 decrees, and
    // perform normalization on the image (basic filtering)
    //THIS LOOKS CORRECT - based on tests
    ImageProcessor imageProcessor =
            new ImageProcessor.Builder()
                    .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                    .add(new ResizeOp(imageSizeX, imageSizeY, ResizeMethod.NEAREST_NEIGHBOR))
                    .add(new Rot90Op(numRoration))
                    .add(getPreprocessNormalizeOp())
                    .build();
    return imageProcessor.process(inputImageBuffer);
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
      pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue(), null));
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