package com.edu.usquat.Classifier;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.media.MediaMetadataRetriever;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;

import com.edu.usquat.R;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.logging.Logger;

import wseemann.media.FFmpegMediaMetadataRetriever;

public class CameraActivity extends Activity implements OnDataPass {
    private static final String TAG = "CameraActivity";
    private static final int PERMISSION_REQUEST = 1;
    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private final int imgSize = 380;
    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        Log.d(TAG,"on create" + this);
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_camera);


        if ( null == savedInstanceState){
            getFragmentManager().beginTransaction()
                    .replace(R.id.container, Camera2VideoFragment.newInstance())
                    .commit();
        }



    }
    // This method is used to get the stored video_path from CameraFragment.

    //KELLY fix so that for 1st version of app -- 2 options: 1) take 40 frames across video 2) sliding window of 40 frames
    // Because of taking approx 1 min to process each set of 40 frames through LSTM, version 1.0 of USquat will only do
    // option 1
    /**
     * USquat method to "recieve" (stored on device) video data and it extracts 40 frames
     * and processes it using the ClassifierActivity that it launches
     * -- 2 options: 1) OPTION GLOBAL_SAMPLE take 40 frames across video
     *               2) OPTION SLIDING_WINDOW sliding window of 40 frames
     *     // Because of taking approx 1 min to process each set of 40 frames through LSTM, version 1.0 of USquat will only do
     *     // option 1
     * @param data
     */
    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    public void onDataPass(String data, String option) {
        Log.d(TAG,data);

        MediaMetadataRetriever fmpeg = new MediaMetadataRetriever();
        fmpeg.setDataSource(data);
        ArrayList<Bitmap> frames = new ArrayList<Bitmap>();
        MediaPlayer mp = MediaPlayer.create(getBaseContext(), Uri.parse(data));
        int microseconds = mp.getDuration() * 1000;  //milliseconds time of video
        //assume average framerate of 30 -- future work to query device,but, changes constantly
        int frameRate = 30;

        //get number of frames
       // mp.getMetrics().get(MediaPlayer.MetricsConstants.FRAMES);
       // int numFrames = mp.getMetrics().get(MediaPlayer.MetricsConstants.FRAMES);

        if(option =="GLOBAL_SAMPLE"){  //new version using instead the FFmpegMediaMetadataRetriever
            //KELLY NEED TO FIX THIS
            //using MediaMetadataRetrivier -which works in microsecond units
            FFmpegMediaMetadataRetriever mmr = new FFmpegMediaMetadataRetriever();
            mmr.setDataSource(getBaseContext(), Uri.parse(data));
            //will need to rotate captured frame
            Matrix matrix = new Matrix();
            matrix.preRotate(-90.0f);
            Bitmap bitmap, resizedBitmap;

            long step = Math.round(1000*1000/frameRate);  //--this is #microseconds to capture 1 frame --mkae steps in microsecond (#microseconds per frame)
            for(long i = 1000000;i<microseconds;i+= step){   // ignoring the first second, grabbing every frame
                // the MediaMetadataRetriever.getFrameAtTime() takes in microseconds 10^-6
                bitmap = mmr.getFrameAtTime(i, FFmpegMediaMetadataRetriever.OPTION_CLOSEST);
             //   Bitmap bitmap = fmpeg.getFrameAtTime(i, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
                resizedBitmap = getResizeBitmap(bitmap,imgSize);
                resizedBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888,true);
                //this FrameGrabber FFmpegMediaMetadataRetriever seems to open the video in rotated mode so rotate -90 degrees

                resizedBitmap = Bitmap.createBitmap(resizedBitmap, 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight(), matrix, true);
                frames.add(resizedBitmap);
            }

        }
        else if(option =="GLOBAL_SAMPLE ORIGINAL"){
            //KELLY NEED TO FIX THIS
            //using MediaMetadataRetrivier -which works in microsecond units
            long step = Math.round(1000*1000/frameRate);  //--this is #microseconds to capture 1 frame --mkae steps in microsecond (#microseconds per frame)
            for(long i = 1000000;i<microseconds;i+= step){   // ignoring the first second, grabbing every frame
                // the MediaMetadataRetriever.getFrameAtTime() takes in microseconds 10^-6
                Bitmap bitmap = fmpeg.getFrameAtTime(i, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
                Bitmap resizedBitmap = getResizeBitmap(bitmap,imgSize);
                resizedBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888,true);
                frames.add(resizedBitmap);
            }

        }
        else {  //SLIDING_WINDOW  -- KELLY FIX or for now just leaving it alone
            //using MediaMetadataRetrivier -which works in microsecond units
            long step = Math.round(1000*1000/frameRate);  //#microseconds to take 1 frame, mkae steps in microsecond (#microseconds per frame)
            for(int i = 1000000;i<microseconds;i+= step){   // ignoring the first second, grabbing every frame
                // the MediaMetadataRetriever.getFrameAtTime() takes in microseconds 10^-6
                Bitmap bitmap = fmpeg.getFrameAtTime(i, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
                Bitmap resizedBitmap = getResizeBitmap(bitmap,imgSize);
                resizedBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888,true);
                frames.add(resizedBitmap);
            }

        }

        Log.d(TAG,String.valueOf(frames.size()));
        Intent intent = new Intent(CameraActivity.this,ClassifierActivity.class);
        BitmapDTO.getInstance().setBitmaps(frames);
        startActivity(intent);
        //processing();
    }


    /**
     * Reduce the size of image
     * @param image
     * @param imgSize
     * @return the resizedBitmap*/
    public Bitmap getResizeBitmap(Bitmap image, int imgSize){
        int width = image.getWidth();
        int height = image.getHeight();

        width = imgSize;
        height = imgSize;
        return Bitmap.createScaledBitmap(image,width,height,true);
    }


}
