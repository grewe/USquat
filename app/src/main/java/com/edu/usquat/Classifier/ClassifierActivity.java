package com.edu.usquat.Classifier;

import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.util.TypedValue;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.VideoView;

import androidx.appcompat.app.AppCompatActivity;

import com.edu.usquat.R;

import java.io.IOException;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import com.edu.usquat.Classifier.Classifier.Device;

/* This Activity handles all results from Classifier Class and display corresponding video to users.
* */
public class ClassifierActivity extends AppCompatActivity {
    private List<Bitmap> frames;
    private final String TAG = "ClassifierActivity";
    private Classifier classifier;
    public List<Classifier.Recognition> temp;
    final Device device = Device.CPU;
    final int numThreads = 1;

    private static String VIDEO_SAMPLE;
    private VideoView mVideoView;
    private TextView mTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
       setContentView(R.layout.classification_layout);
        frames = BitmapDTO.getInstance().getBitmaps();
        Log.d(TAG,String.valueOf(frames.size()));
        recreateClassifier(device, numThreads);
        if (classifier == null) {
            Log.e(TAG,"No classifier on preview!");
            return;
        }
        processing();
        mVideoView = findViewById(R.id.video_view);
        mTextView = findViewById(R.id.textView3);
    }
    @Override
    protected void onStart() {
        super.onStart();
        initializePlayer();
    }

    @Override
    protected void onStop(){
        super.onStop();
        releasePlayer();
    }

    private void recreateClassifier(Device device, int numThreads) {
        if (classifier != null) {
            Log.d(TAG,"Closing classifier.");
            classifier.close();
            classifier = null;
        }
        try {
            Log.d(TAG,
                    String.format("Creating classifier (device=%s, numThreads=%d)",device,numThreads));
            classifier = Classifier.create(this, device, numThreads,getAssets());

        } catch (IOException e) {
            Log.e(TAG,String.valueOf(e));
            Log.e(TAG, "Failed to create classifier.");
        }


    }

    protected void processing(){
        if (classifier != null){
          temp = classifier.getFrames(frames);
          Log.d(TAG,String.valueOf(temp));
        }

    }

    private Uri getMedia(String mediaName){
        return Uri.parse("android.resource://" + getPackageName() + "/raw/" + mediaName);
    }

    private void initializePlayer(){
        String result = String.valueOf(temp.get(0));
        String[] detected = result.split("\\s+");
        Log.d(TAG, detected[1]);
        mTextView.setTextSize(TypedValue.COMPLEX_UNIT_DIP,14);
        mTextView.setText(detected[1]);
        switch(detected[1]){
            case "shallow":
                VIDEO_SAMPLE = "shallow";
                break;
            case "good":
                VIDEO_SAMPLE = "good";
                break;
            case "heels_off":
                VIDEO_SAMPLE = "heels_off";
                break;
            case "bent_over":
                VIDEO_SAMPLE = "bent_over";
                break;
            case "knees_in":
                VIDEO_SAMPLE = "knees_in";
                break;
        }
        Uri videoUri = getMedia(VIDEO_SAMPLE);
        mVideoView.setVideoURI(videoUri);
        mVideoView.start();
    }

    private void releasePlayer(){
        mVideoView.stopPlayback();
    }
}
