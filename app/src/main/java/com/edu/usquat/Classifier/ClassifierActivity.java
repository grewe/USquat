package com.edu.usquat.Classifier;

import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;
import android.widget.VideoView;

import androidx.appcompat.app.AppCompatActivity;

import com.edu.usquat.R;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import com.edu.usquat.Classifier.Classifier.Device;

public class ClassifierActivity extends AppCompatActivity {
    private List<Bitmap> frames;
    private final String TAG = "ClassifierActivity";
    private Classifier classifier;
    public List<Classifier.Recognition> temp;
    final Device device = Device.CPU;
    final int numThreads = 1;

    private static final String VIDEO_SAMPLE = "good";
    private VideoView mVideoView;

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
//        runOnUiThread(
//                new Runnable() {
//                    @Override
//                    public void run() {
//                        if (temp != null){
//                            Log.d(TAG,String.valueOf(temp));
//                        }
//                    }
//                }
//        );

    }

    private Uri getMedia(String mediaName){
        return Uri.parse("android.resource://" + getPackageName() + "/raw" + mediaName);
    }

    private void initializePlayer(){
        Uri videoUri = getMedia(VIDEO_SAMPLE);
        mVideoView.setVideoURI(videoUri);
        mVideoView.start();
    }

    private void releasePlayer(){
        mVideoView.stopPlayback();
    }
}
