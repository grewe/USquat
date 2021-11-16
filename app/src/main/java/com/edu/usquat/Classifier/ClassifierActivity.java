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
    final int numThreads = 2;



    private static String VIDEO_SAMPLE;
    private VideoView mVideoView;
    private TextView mTextView;
    Boolean       FLAG_PROCESSING_DONE;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
       setContentView(R.layout.classification_layout);


        frames = BitmapDTO.getInstance().getBitmaps();
        Log.d(TAG,String.valueOf(frames.size()));

        //create the classifier model
        recreateClassifier(device, numThreads);
        if (classifier == null) {
            Log.e(TAG,"No classifier on preview!");
            return;
        }

        //get handles to reporting text area (mTextView) to report classification results
        // and handle to video display view if corrrectie video is necessary (not a good squat)
        mVideoView = findViewById(R.id.video_view);
        mTextView = findViewById(R.id.classification_message_text_view);


        //tell user to wait while processing video
        mTextView.setText("PLEASE WAIT");



        //start processing - grab 40(x) frames, sent to feature extractor then to the USquat Classifier in processing()
        FLAG_PROCESSING_DONE = false;
        new Thread(){
            @Override
            public void run() {
                super.run();
                processing();
            }
        }.start();

       // processing();
//        mVideoView = findViewById(R.id.video_view);
//        mTextView = findViewById(R.id.classification_message_text_view);
//        mTextView.setText("PLEASE WAIT");
       // processing();
    }
    @Override
    protected void onStart() {
        super.onStart();
//        initializePlayer();

    }

    @Override
    protected void onStop(){
        super.onStop();
        releasePlayer();
    }

    //create classifier (if already open, close and recreate)
    private void recreateClassifier(Device device, int numThreads) {
        if (classifier != null) {
            Log.d(TAG,"Closing classifier.");
            classifier.close();
            classifier = null;
        }
        try {
            Log.d(TAG,
                    String.format("Creating classifier (device=%s, numThreads=%d)",device,numThreads));
            //create classifier
            classifier = Classifier.create(this, device, numThreads,getAssets());

        } catch (IOException e) {
            Log.e(TAG,String.valueOf(e));
            Log.e(TAG, "Failed to create classifier.");
        }


    }

    //MAIN processing method
    protected void processing(){
        if (classifier != null){
            // Processing the frames
          temp = classifier.getFramesAndProcess(frames);
          Log.d(TAG,String.valueOf(temp));
        }
        FLAG_PROCESSING_DONE= true;
       this.runOnUiThread(new Runnable() {
           @Override
           public void run() {
               initializePlayer();
           }
       });

    }

    private Uri getMedia(String mediaName){
        return Uri.parse("android.resource://" + getPackageName() + "/raw/" + mediaName);
    }

    private void initializePlayer(){
        String result = String.valueOf(temp.get(0));
        String[] detected = result.split("\\s+");
        Log.d(TAG, detected[1]);
        mTextView.setTextSize(TypedValue.COMPLEX_UNIT_DIP,28);

        String message ="Your squat ";
        switch(detected[1]){
            case "shallow":
                VIDEO_SAMPLE = "shallow";
                message += " is too shallow";
                break;
            case "good":
                VIDEO_SAMPLE = "good";
                message += "is good";
                break;
            case "heels_off":
                VIDEO_SAMPLE = "heels_off";
                message = "Your heels are off the ground";
                break;
            case "bent_over":
                VIDEO_SAMPLE = "bent_over";
                message ="You are too bent over";
                break;
            case "knees_in":
                VIDEO_SAMPLE = "knees_in";
                message = "Your knees are collapsing in";
                break;
        }
        mTextView.setText(message);
        Uri videoUri = getMedia(VIDEO_SAMPLE);
        mVideoView.setVideoURI(videoUri);
        mVideoView.start();
    }

    private void releasePlayer(){
        mVideoView.stopPlayback();
    }
}
