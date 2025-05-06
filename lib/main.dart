import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'dart:math';
import 'package:image/image.dart' as img;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pill Recognition',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // TRY THIS: Try running your application with "flutter run". You'll see
        // the application has a purple toolbar. Then, without quitting the app,
        // try changing the seedColor in the colorScheme below to Colors.green
        // and then invoke "hot reload" (save your changes or press the "hot
        // reload" button in a Flutter-supported IDE, or press "r" if you used
        // the command line to start the app).
        //
        // Notice that the counter didn't reset back to zero; the application
        // state is not lost during the reload. To reset the state, use hot
        // restart instead.
        //
        // This works for code too, not just values: Most code changes can be
        // tested with just a hot reload.
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.yellow),
        useMaterial3: true,
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  XFile? image;

  dynamic _probability = 0;
  String? _result;
  List<String>? _labels;
  late tfl.Interpreter _interpreter;

  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Scaffold(
      appBar: AppBar(
        // TRY THIS: Try changing the color here to a specific color (to
        // Colors.amber, perhaps?) and trigger a hot reload to see the AppBar
        // change color while the other colors stay the same.
        backgroundColor: Colors.white,
        // Here we take the value from the MyHomePage object that was created by
        // the App.build method, and use it to set our appbar title.
        title: const Text(
          'Pill Recognition',
          style: TextStyle(
            color: Colors.black,
            fontWeight: FontWeight.bold,
          ),
        ),
        centerTitle: true,
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              children: [
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.yellow,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 30, vertical: 15),
                  ),
                  onPressed: () async {
                    final image = await ImagePicker()
                        .pickImage(source: ImageSource.camera);
                    if (image != null) {
                      setState(() {
                        this.image = image;
                        runInference();
                      });
                    }
                  },
                  child: const Text(
                    '+Take a photo',
                    style: TextStyle(
                      color: Colors.black,
                      fontSize: 16,
                    ),
                  ),
                ),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.yellow,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 30, vertical: 15),
                  ),
                  onPressed: () async {
                    final image = await ImagePicker()
                        .pickImage(source: ImageSource.gallery);
                    if (image != null) {
                      setState(() {
                        this.image = image;
                        runInference();
                      });
                    }
                  },
                  child: const Text(
                    '+ Photo album',
                    style: TextStyle(
                      color: Colors.black,
                      fontSize: 16,
                    ),
                  ),
                ),
              ],
            ),
          ),
          if (image != null)
            AspectRatio(
              aspectRatio: 1,
              child: Image.file(
                fit: BoxFit.cover,
                File(image!.path),
                width: MediaQuery.of(context).size.width,
              ),
            ),

          if (_result != null)
            Text(
              'Result:  $_result',
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
        ],
      ),
    );
  }

  Future<void> loadModel() async {
    try {
      // Try model6.tflite first
      try {
        _interpreter = await tfl.Interpreter.fromAsset('assets/model6.tflite');

        // Print model input/output shape information
        var inputShape = _interpreter.getInputTensor(0).shape;
        var outputShape = _interpreter.getOutputTensor(0).shape;
        debugPrint('Model6 input shape: $inputShape');
        debugPrint('Model6 output shape: $outputShape');
      } catch (e) {
        // If model6 fails, fall back to model1
        debugPrint('Error loading model6: $e. Trying model1...');
        _interpreter = await tfl.Interpreter.fromAsset('assets/model1.tflite');

        // Print model input/output shape information
        var inputShape = _interpreter.getInputTensor(0).shape;
        var outputShape = _interpreter.getOutputTensor(0).shape;
        debugPrint('Model1 input shape: $inputShape');
        debugPrint('Model1 output shape: $outputShape');
      }
    } catch (e) {
      debugPrint('Error loading models: $e');
    }
  }

  Future<Uint8List> preprocessImage(File imageFile) async {
    // Decode the image to an Image object
    img.Image? originalImage = img.decodeImage(await imageFile.readAsBytes());

    // Resize the image to 96x96 to match model6 input shape
    img.Image resizedImage =
        img.copyResize(originalImage!, width: 96, height: 96);

    Uint8List bytes = resizedImage.getBytes();

    if (bytes.length != 96 * 96 * 3) {
      List<int> rgbOnly = [];
      for (int i = 0; i < bytes.length; i += 4) {
        if (i + 2 < bytes.length) {
          rgbOnly.add(bytes[i]); // R
          rgbOnly.add(bytes[i + 1]); // G
          rgbOnly.add(bytes[i + 2]); // B
          // Skip Alpha channel
        }
      }
      bytes = Uint8List.fromList(rgbOnly);
    }

    return bytes;
  }

  Future<void> runInference() async {
    if (_labels == null) {
      return;
    }

    try {
      debugPrint('Starting inference process');
      Uint8List inputBytes = await preprocessImage(File(image!.path));
      debugPrint('Image preprocessed, size: ${inputBytes.length}');

      if (inputBytes.length != 96 * 96 * 3) {
        debugPrint(
            'Warning: Image size mismatch. Expected ${96 * 96 * 3}, got ${inputBytes.length}');
      }

      Int8List int8Input = Int8List(inputBytes.length);
      for (int i = 0; i < inputBytes.length; i++) {
        int8Input[i] = (inputBytes[i] - 128).toInt().clamp(-128, 127);
      }
      debugPrint('Converted uint8 to int8 range (-128 to 127)');

      // Reshape to 96x96 image dimensions to match the model's input shape
      var input = int8Input.reshape([1, 96, 96, 3]);
      debugPrint('Input tensor shaped to [1, 96, 96, 3] as int8');

      var outputBuffer = List<int>.filled(1 * 2, 0).reshape([1, 2]);
      debugPrint('Output buffer created with shape [1, 2] as int');

      try {
        debugPrint('Running TensorFlow inference...');
        _interpreter.run(input, outputBuffer);
        debugPrint('Inference completed successfully');

        List<int> outputList = [];
        if (outputBuffer[0] is List<dynamic>) {
          for (var item in outputBuffer[0]) {
            outputList.add(item as int);
          }
        } else {
          outputList = List<int>.from(outputBuffer[0]);
        }

        List<double> output = outputList.map((e) => e.toDouble()).toList();

        // Print raw output for debugging
        debugPrint('Raw output integers: $outputList');
        debugPrint('Converted to double: $output');

        // Calculate probability
        double maxScore = output.reduce(max);
        _probability = maxScore / 100.0; 

        // Get the classification result
        int highestProbIndex = output.indexOf(maxScore);
        String classificationResult = _labels![highestProbIndex];
        debugPrint(
            'Classification result: $classificationResult, score: $maxScore');

        setState(() {
          _result = classificationResult;
          // _probability is updated with the calculated probability
        });

        navigateToResult();
      } catch (inferenceError) {
        debugPrint('TensorFlow inference error: $inferenceError');
        // Try with a different approach
        debugPrint('Trying alternative inference approach...');
        var inputs = [input];
        var outputs = {0: outputBuffer};
        _interpreter.runForMultipleInputs(inputs, outputs);
        debugPrint('Alternative inference completed');

        var altOutput = outputs[0];
        if (altOutput != null) {
          List<int> outputList = [];
          for (var item in (altOutput as List)[0]) {
            outputList.add(item as int);
          }

          List<double> output = outputList.map((e) => e.toDouble()).toList();

          double maxScore = output.reduce(max);
          _probability = maxScore / 100.0;
          int highestProbIndex = output.indexOf(maxScore);
          String classificationResult = _labels![highestProbIndex];

          setState(() {
            _result = classificationResult;
          });

          navigateToResult();
        }
      }
    } catch (e) {
      debugPrint('Error during inference: $e');
    }
  }

  Future<List<String>> loadLabels() async {
    final labelsData = await DefaultAssetBundle.of(context)
        .loadString('assets/archi_label.txt');
    return labelsData.split('\n');
  }

  String classifyImage(List<int> output) {
    int highestProbIndex = output.indexOf(output.reduce(max));
    return _labels![highestProbIndex];
  }

  void navigateToResult() {
    
    // Navigator.push(
    //   context,
    //   MaterialPageRoute(
    //     builder: (context) => ResultPage(
    //       image: _image,
    //       result: _result!,
    //       probability: _probability,
    //     ),
    //   ),
    // );
  }

  @override
  void initState() {
    super.initState();
    loadModel().then((_) {
      loadLabels().then((loadedLabels) {
        setState(() {
          _labels = loadedLabels;
        });
      });
    });
  }

  @override
  void dispose() {
    super.dispose();
    _interpreter.close();
  }
}
