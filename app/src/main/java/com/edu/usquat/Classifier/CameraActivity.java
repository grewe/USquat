package com.edu.usquat.Classifier;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
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
    @Override
    public void onDataPass(String data) {
        Log.d(TAG,data);

        MediaMetadataRetriever fmpeg = new MediaMetadataRetriever();
        fmpeg.setDataSource(data);
        ArrayList<Bitmap> frames = new ArrayList<Bitmap>();
        MediaPlayer mp = MediaPlayer.create(getBaseContext(), Uri.parse(data));
        int seconds = mp.getDuration() * 1000;
        int frameRate = 30;
        long step = Math.round(1000*1000/frameRate);
        for(int i = 1000000;i<seconds;i+= step){
            Bitmap bitmap = fmpeg.getFrameAtTime(i, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
            Bitmap resizedBitmap = getResizeBitmap(bitmap,imgSize);
            resizedBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888,true);
            frames.add(resizedBitmap);
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
