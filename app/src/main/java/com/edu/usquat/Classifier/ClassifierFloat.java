package com.edu.usquat.Classifier;

import android.app.Activity;
import android.content.res.AssetManager;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

import java.io.IOException;

public class ClassifierFloat extends Classifier {

    /** Float MobileNet requires additional normalization of the used input. */
    private static final float IMAGE_MEAN = 127.5f;

    private static final float IMAGE_STD = 127.5f;

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static final float PROBABILITY_MEAN = 0.0f;

    private static final float PROBABILITY_STD = 1.0f;

    /**
     * Initializes a {@code ClassifierFloatMobileNet}.
     *
     * @param activity
     */
    public ClassifierFloat(Activity activity, Device device, int numThreads, AssetManager assetManager)
            throws IOException {
        super(activity, device, numThreads,assetManager);
    }

    // TODO: Specify model.tflite as the model file and labels.txt as the label file

    @Override
    protected String getModelPath() {
        return "feature_extraction_with_OP.tflite";
    }


    @Override
    protected String getLabelPath() {
        return "labels.txt";
    }


    @Override
    protected TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    @Override
    protected TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }
}
