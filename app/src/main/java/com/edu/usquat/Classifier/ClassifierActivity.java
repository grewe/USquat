package com.edu.usquat.Classifier;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Parcelable;
import android.util.Log;
import android.util.TypedValue;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.VideoView;
//import android.widget.ProgressBar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.gridlayout.widget.GridLayout;

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
public class ClassifierActivity<ProgessBar> extends AppCompatActivity {
    public List<Bitmap> frames;
    private final String TAG = "ClassifierActivity";
    private Classifier classifier;
    public List<Classifier.Recognition> results;
    final Device device = Device.CPU;
    final int numThreads = 2;
    public Switch mode;
    private static String VIDEO_SAMPLE;
    private VideoView mVideoView;
    private TextView mTextView;
    Boolean       FLAG_PROCESSING_DONE;
    String message;
//ProgressBar pg;


    @SuppressLint("WrongViewCast")
    @Override

    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
      //  pg = (ProgressBar) findViewById(R.id.progressBar);
       setContentView(R.layout.classification_layout);
        frames = BitmapDTO.getInstance().getBitmaps();
        Log.d(TAG,String.valueOf(frames.size()));
        recreateClassifier(device, numThreads);
        if (classifier == null) {
            Log.e(TAG,"No classifier on preview!");
            return;
        }
        mVideoView = findViewById(R.id.video_view);
        mTextView = findViewById(R.id.classification_message_text_view);
       // pg.setVisibility(View.VISIBLE);
        mTextView.setText("PLEASE WAIT");
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

    /* This will call getFramesAndProcess from Classifier and return the recognition results
    * */
    protected void processing(){

        if (classifier != null){
            // Processing the frames
            // result will look like this - priority queue [shallow] 99.5 [bent_over] 0.1

     results = classifier.getFramesAndProcess(frames);

          Log.d(TAG,String.valueOf(results));
            Log.d(TAG,String.valueOf(String.format("%1$TH:%1$TM:%1$TS",System.currentTimeMillis())));
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
        message ="Your squat ";

        String result = String.valueOf(results.get(0));
        String[] detected = result.split("\\s+");
        Log.d(TAG, detected[1]);
        mTextView.setTextSize(TypedValue.COMPLEX_UNIT_DIP,28);


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
// Added code for switch here
        // keep code in if statement intact since that was Kelly's code
        mode = (Switch) findViewById(R.id.switch1);
        mode.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {

                if (isChecked){

                 /*  Intent intent    = new Intent(ClassifierActivity.this, GridAct.class);

                   intent.putExtra("frames", (Parcelable) frames);

                   startActivity(intent);*/



                   BitmapDTO.getInstance().setBitmaps(frames);
                   Intent intent1   = new Intent(ClassifierActivity.this,GridAct.class);
                   startActivity(intent1);

                  /*  int j =0;

                    //startActivity(new Intent(ClassifierActivity.this, GridAct.class));
                    Intent i = new Intent(ClassifierActivity.this, GridAct.class);

                    i.putExtra("Size",frames.size());
                    while(j < frames.size()){
                        i.putExtra("framesI"+j, frames.get(j));
                        j++;

                    }
                    startActivity(i);*/

                    //Bundle bundle = new Bundle();
                    //bundle.putParcelableArray("images",frames);
                    //i.putExtras(bundle);
                }
                else {

        /*GridLayout gridLayout = (GridLayout) findViewById(R.id.imagegrid);

        gridLayout.removeAllViews();


        int total = frames.size();
        int column = 2;
        int row = total / column;
        gridLayout.setColumnCount(column);
        gridLayout.setRowCount(row +1 );
        for (int i = 0, c = 0, r = 0; i < total; i++, c++) {
            if (c == column) {
                c = 0;
                r++;
            }
            ImageView oImageView = new ImageView(ClassifierActivity.this);



            oImageView.setImageBitmap(frames.get(i));



            // oImageView.setImageBitmap(frames.get(4));

            oImageView.setLayoutParams(new GridLayout.LayoutParams());

            GridLayout.Spec rowSpan = GridLayout.spec(GridLayout.UNDEFINED, 1);
            GridLayout.Spec colspan = GridLayout.spec(GridLayout.UNDEFINED, 1);
            if (r == 0 && c == 0) {

                colspan = GridLayout.spec(GridLayout.UNDEFINED, 2);
                rowSpan = GridLayout.spec(GridLayout.UNDEFINED, 2);
            }
            GridLayout.LayoutParams gridParam = new GridLayout.LayoutParams(
                    rowSpan, colspan);
            gridLayout.addView(oImageView, gridParam);
        }*/
        mTextView.setText(message);
        Uri videoUri = getMedia(VIDEO_SAMPLE);
        mVideoView.setVideoURI(videoUri);
        mVideoView.start();
        mVideoView.requestFocus();
        mVideoView.setOnPreparedListener(mp -> mp.setLooping(true));

                }
            }
        });


    }

    private void releasePlayer(){
        mVideoView.stopPlayback();
    }
}
