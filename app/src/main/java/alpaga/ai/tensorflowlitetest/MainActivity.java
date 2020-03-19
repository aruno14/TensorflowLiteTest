package alpaga.ai.tensorflowlitetest;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            MappedByteBuffer tfliteModelCaption = FileUtil.loadMappedFile(this, "classifier_lstm2.tflite");
            Interpreter.Options options = new Interpreter.Options();
            Interpreter tflite = new Interpreter(tfliteModelCaption, options);
            String test = "Test: " + tflite.getInputTensorCount()
                    + " - " + tflite.getInputTensor(0).name() + ":" + Arrays.toString(tflite.getInputTensor(0).shape()) + ":" + tflite.getInputTensor(0).dataType()
                    + " - " + tflite.getInputTensor(1).name() + ":" + Arrays.toString(tflite.getInputTensor(1).shape()) + ":" + tflite.getInputTensor(1).dataType();
            Log.d("Tensorflow", test);
            float[][] inputString = new float[1][10];
            //inputString[0][9] = 4;
            float[][] inputData = new float[1][2];
            //inputData[0][0] = 0;
            //inputData[0][1] = 1;
            Object[] inputArray = {inputString, inputData};
            Map<Integer, Object> outputMap = new HashMap<>();
            outputMap.put(0, new float[1][61]);
            tflite.runForMultipleInputsOutputs(inputArray, outputMap);
            tflite.close();
            float[] result = ((float[][]) outputMap.get(0))[0];
            Log.d("Tensorflow", Arrays.toString(result));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
