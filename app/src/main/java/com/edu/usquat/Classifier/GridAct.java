package com.edu.usquat.Classifier;

import androidx.appcompat.app.AppCompatActivity;
import androidx.gridlayout.widget.GridLayout;

import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Parcelable;
import android.widget.ImageView;

import com.edu.usquat.R;

import java.util.List;

public class GridAct extends AppCompatActivity {
    public static final String TAG = "Intent INput";
    public List<Bitmap> frames;


    int size;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_grid);
        int j = 0;

      frames = BitmapDTO.getInstance().getBitmaps();

     /*   for ( j = 0 ; j < frames1.length; j++)
        {
            frames1[j] ;

        }*/

       /* size = getIntent().getIntExtra("Size", 40);

        while (j < size) {
            frames.set(j, (Bitmap) getIntent().getParcelableExtra("framesI" + j));
            j++;
        }
*/
        GridLayout gridLayout = (GridLayout) findViewById(R.id.imagegrid);

        gridLayout.removeAllViews();


        int total = frames.size();
        int column = 2;
        int row = total / column;
        gridLayout.setColumnCount(column);
        gridLayout.setRowCount(row + 1);
        for (int i = 0, c = 0, r = 0; i < total; i++, c++) {
            if (c == column) {
                c = 0;
                r++;
            }
            ImageView oImageView = new ImageView(GridAct.this);


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
        }
    }
}