package com.edu.usquat;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {
    /**
     * Buttons
     */
    private Button uploadButton;
    private Button cameraButton;

    /**
     * flag for whether we want to run in diagnostic mode or not
     */
    public static boolean DIAGNOSTIC_MODE = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialization
        initViewHooks(); // initialize the hooks to view screen
        initButtonListeners();
    }


    private void initViewHooks(){
        // Button Handler
        this.cameraButton = (Button) findViewById(R.id.camera_button);
    };
    /**
     * initialize on click listeners to all the buttons on screen
     */
    private void initButtonListeners() {
        // Create event handler for START button
        cameraButton.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                Intent intent = new Intent ("com.edu.usquat.localize.ClassifierActivity");
                startActivity(intent);
            }
        });
    };
}