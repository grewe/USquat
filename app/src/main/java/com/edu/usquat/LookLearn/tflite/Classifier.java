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

package com.edu.usquat.LookLearn.tflite;

import android.graphics.Bitmap;
import android.graphics.Point;
import android.graphics.RectF;

import java.util.List;

/** Generic interface for interacting with different recognition engines. */
public interface Classifier {
  List<Recognition> recognizeImage(Bitmap bitmap);

  void enableStatLogging(final boolean debug);

  String getStatString();

  void close();

  void setNumThreads(int num_threads);

  void setUseNNAPI(boolean isChecked);

  /** An immutable result returned by a Classifier describing what was recognized. */
  public class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /**
     * Display name for the recognition.
     */
    private String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private  Float confidence;

    /**
     * Optional location within the source image for the location of the recognized object.
     */
    private RectF location;

    /**
     * maximum temperature found in bounding box (used ONLY in IR) represented in Celcius units
     */
    private double maxTemp;


    /**
     * location (x,y) of maximum temperature
     */
    private Point maxTempLocation;

    public Recognition(
            final String id, final String title, final Float confidence, final RectF location) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
      this.location = location;
      this.maxTemp=maxTemp;
      this.maxTempLocation=maxTempLocation;
    }

    /**
     * construtor that takes as input another recognition object
     * @param r
     */
    public Recognition(Recognition r){
      this.id = r.id;
      this.title = r.title;
      this.location  = new RectF(r.getLocation());
      this.confidence  = r.confidence;
      this.maxTemp = r.maxTemp;
      if(r.maxTempLocation != null)
        this.maxTempLocation = new Point(r.maxTempLocation);

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

    public double getMaxTemp() {
      return  maxTemp;
    }

    public void setMaxTemp(double maxTemp) {
      this.maxTemp = maxTemp;
    }

    public void setMaxTempLocation(Point maxTempLocation) {
      this.maxTempLocation = maxTempLocation;
    }

    public void setConfidence(float  c) {
      this.confidence =c;
    }
    public void setTitle(String t){
      this.title = t;
    }

    public Point getMaxTempLocation() {
      return new Point(maxTempLocation);
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
}
