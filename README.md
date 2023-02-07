# gmshimg
A simple GUI & AI model to recognize simple geometrical objects(circle, rectangle, triangle).

Requirements
------------
* A suitable [pipenv](https://pipenv.pypa.io/en/latest/) environment can be created and activated with:

```
pipenv install
pipenv shell
```
* Cmake, C++ compiler, Flutter

## Installation
This will guide through steps: clone repo; generate dataset; train model using that dataset; convert trained model to tflite format; build gui.

### Clone repo

```
git clone https://github.com/vnepogodin/gmshimg.git
cd gmshimg
```

### Generation of dataset
To generate dataset you need to run these commands:

1) Generating images for dataset
```
cd tools
python generate_images_for_dataset.py
tar -czf geometric_shapes_images.tar.gz geometric_shapes_images
```

2) Using the generated images to create dataset
```
cd ../my_dataset
mkdir -p ~/tensorflow_datasets/downloads/manual/
cp ../tools/geometric_shapes_images.tar.gz ~/tensorflow_datasets/downloads/manual/
tfds build
```

### Training of the model using script
To train model you need to run these commands:

1) Training of the model
```
cd ..
python train_model.py
```

2) Run prediction with trained model
```
python dataset_test.py
```

### Using existing datasets

- https://www.kaggle.com/datasets/smeschke/four-shapes

- https://www.kaggle.com/datasets/cactus3/basicshapes

### Training of the model using Teachable Machine

Follow these steps:

1) Go to https://teachablemachine.withgoogle.com and click **Get Started**.

2) Choose **Standard Image Model**.

3) Add the classes and edit the labels of each class.

create labels in this order: "circle", "rectangle", "triangle".

4) Add your images for the dataset by clicking **Upload** under each class

5) Click **Train Model** to train the mode.

6) After the training completes, test the model with other plant images (optional).

7) Export the model by clicking **Export Model** on the Preview panel, and click **Download model**.

The downloaded model will contain: model (*.h5), file containing labels (labels.txt).

Rename downloaded model (*.h5) to gmshimg.h5.

### Convert model
You need to run this command to convert .h5 format model to .tflite,
for mobile and GUI use.

```
python convert_model_to_tflite.py
```


### Building tensorflow lite fron source (for Windows, Linux, MacOS)

1) Clone tensorflow repository
```
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

2) Create build folder
```
mkdir tflite_build
cd tflite_build
```

3) Build in release mode
```
cmake ../tensorflow_src/tensorflow/lite
cmake --build . -j
```

4) Copy the built library
```
cd ../tutapp
mkdir blobs
cp * blobs/
```

### Download tensorflow lite (for Android)

###### On Windows use `.\install.bat` instead of `./install.sh`.

```
cd tutapp
./install.sh
```

### Build and Run GUI


```
cd tutapp
flutter run
```


### Why did I do that? (yet another AI?)

I made that project to learn how to develop AI,
learn how to work with tensorflow framework; make a dataset for my model.
So that's about it ;)

### Credits

* https://www.tensorflow.org/lite/guide/reduce_binary_size.
* https://www.tensorflow.org/datasets/catalog/eurosat.
* https://www.tensorflow.org/datasets/overview.
* https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/mnist.py.
* https://www.tensorflow.org/tutorials/images/classification.
* https://pub.dev/packages/tflite_flutter.
