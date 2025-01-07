package com.google_mlkit_pose_detection;

import android.content.AsyncQueryHandler;
import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.pose.PoseDetection;
import com.google.mlkit.vision.pose.PoseLandmark;
import com.google.mlkit.vision.pose.accurate.AccuratePoseDetectorOptions;
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions;
import com.google_mlkit_commons.InputImageConverter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import okhttp3.internal.Util;

public class PoseDetector implements MethodChannel.MethodCallHandler {
    private static final String START = "vision#startPoseDetector";
    private static final String CLOSE = "vision#closePoseDetector";

    private final Context context;
    private final Map<String, com.google.mlkit.vision.pose.PoseDetector> instances = new HashMap<>();
    Executor executor = new ThreadPoolExecutor(
            6,
            18,
            1000,
            TimeUnit.MILLISECONDS,
            new ArrayBlockingQueue<>(4),
            new BackgroundThreadFactory("pose_executor"));

    public PoseDetector(Context context) {
        this.context = context;
    }

    @Override
    public void onMethodCall(@NonNull MethodCall call, @NonNull MethodChannel.Result result) {
        String method = call.method;
        switch (method) {
            case START:
                handleDetection(call, result);
                break;
            case CLOSE:
                closeDetector(call);
                result.success(null);
                break;
            default:
                result.notImplemented();
                break;
        }
    }

    private void handleDetection(MethodCall call, final MethodChannel.Result result) {
        Map<String, Object> imageData = (Map<String, Object>) call.argument("imageData");
        InputImage inputImage = InputImageConverter.getInputImageFromData(imageData, context, result);
        if (inputImage == null) return;

        String id = call.argument("id");
        com.google.mlkit.vision.pose.PoseDetector poseDetector = instances.get(id);
        if (poseDetector == null) {
            Map<String, Object> options = call.argument("options");
            if (options == null) {
                result.error("PoseDetectorError", "Invalid options", null);
                return;
            }

            String mode = (String) options.get("mode");
            int detectorMode = PoseDetectorOptions.STREAM_MODE;
            if (mode.equals("single")) {
                detectorMode = PoseDetectorOptions.SINGLE_IMAGE_MODE;
            }

            String model = (String) options.get("model");
            if (model.equals("base")) {
                PoseDetectorOptions detectorOptions = new PoseDetectorOptions.Builder()
                        .setDetectorMode(detectorMode)
                        .setExecutor(executor)
                        .build();
                poseDetector = PoseDetection.getClient(detectorOptions);
            } else {
                AccuratePoseDetectorOptions detectorOptions = new AccuratePoseDetectorOptions.Builder()
                        .setDetectorMode(detectorMode)
                        .setExecutor(executor)
                        .build();
                poseDetector = PoseDetection.getClient(detectorOptions);
            }
            instances.put(id, poseDetector);
        }

        poseDetector.process(inputImage)
                .addOnSuccessListener(
                        pose -> {
                            List<List<Map<String, Object>>> array = new ArrayList<>();
                            if (!pose.getAllPoseLandmarks().isEmpty()) {
                                List<Map<String, Object>> landmarks = new ArrayList<>();
                                for (PoseLandmark poseLandmark : pose.getAllPoseLandmarks()) {
                                    Map<String, Object> landmarkMap = new HashMap<>();
                                    landmarkMap.put("type", poseLandmark.getLandmarkType());
                                    landmarkMap.put("x", poseLandmark.getPosition3D().getX());
                                    landmarkMap.put("y", poseLandmark.getPosition3D().getY());
                                    landmarkMap.put("z", poseLandmark.getPosition3D().getZ());
                                    landmarkMap.put("likelihood", poseLandmark.getInFrameLikelihood());
                                    landmarks.add(landmarkMap);
                                }
                                array.add(landmarks);
                            }
                            result.success(array);
                        })
                .addOnFailureListener(
                        e -> result.error("PoseDetectorError", e.toString(), null));
    }

    private void closeDetector(MethodCall call) {
        String id = call.argument("id");
        com.google.mlkit.vision.pose.PoseDetector poseDetector = instances.get(id);
        if (poseDetector == null) return;
        poseDetector.close();
        instances.remove(id);
    }
}

class BackgroundThreadFactory implements ThreadFactory {
    private String tag;

    BackgroundThreadFactory(String tag) {
        this.tag = tag;
    }

    @Override
    public Thread newThread(Runnable runnable) {
        Thread thread = new Thread(runnable, "CustomThread" + tag);
        thread.setName("CustomThread" + tag);
        thread.setPriority(Thread.MAX_PRIORITY);

        // A exception handler is created to log the exception from threads
        thread.setUncaughtExceptionHandler(new Thread.UncaughtExceptionHandler() {
            @Override
            public void uncaughtException(Thread thread, Throwable ex) {
                Log.e("Thread error", thread.getName() + " encountered an error: " + ex.getMessage());
            }
        });
        return thread;
    }
}