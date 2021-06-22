# Garbage-Classification-with-ANN-Model-Project
Data set link: https://www.kaggle.com/asdasdasasdas/garbage-classification
As seen in the image, it contains 6 different waste groups (cardboard, glass, metal, paper, plastic and garbage).
A dataset has been created for you. This data set is divided into train and test. From you
You are expected to do the following:
1- The data set is already divided into train and test. To train and test the model you will create
to read the data. Reserve 10% of train data for validation.

2- To solve the above problem, an artificial model consisting of fully dependent layers
Build a neural network model. The model you created is shared using the appropriate parameters.
Train with the dataset. (explain the model summary, code, parameters used)
add screenshot)
3- Train accuracy validation accuracy, train loss and validation loss resulting from the training
Add the image of draw curves here.
4- Test your training result model with test data, Accuracy, Recall, Precision and F1 score
Calculate the values. Calculate the complex matrix and add an image here.
5- Interpret the results of your model, if it is unsuccessful, why is it successful?

ImageDataGenerator is the class Keras refers to for pipelined editing of image data for deep learning. It provides easy access to our local file system and multiple different ways to load data from different structures.
Here are the methods I use for train_datagen:
Rescale: Rescale factor. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided. (after applying all other transformations)
Shear_range: Shear Density (Anti-clockwise shear angle in degrees) I set to 0.1.
Zoom_range : Random zoom range. I set it to 0.1.
Width_shift_range: Float, int - float: fraction of total width if <1 or pixels if >=1. are random items from the array. I set the fraction of the width to 0.1.
Height_shift_range: We can say the fraction of the height or the pixel sizing. I set the fraction of the height to 0.1.
Horizontal_flip: Takes bool values. It means randomly flipping the inputs horizontally.
Vertical_flip: Takes bool values. It means randomly flipping the inputs vertically.
Validation_split: The fraction of images reserved for validation (strictly between 0 and 1). We were asked to separate it as 0.1, so I separated our validation data set as 0.1.

The methods I use in the train_datagen.Flow_from_directory function:
Directory, that is, the directory where the base_path train data set I gave as the directory is located.
Target_size: tuple of integers (height, width), default: (256, 256). All found images will be resized. I set target_size to (100,100).
Batch_size: I set the size of the data chunks to 32.
Class_mode: "binary", "categorical", "input", "multi_output". The default is "categorical" and I have defined class_mode as "categorical" which should be valid for our project. Mode of exporting targets: - "binary": 1D numeric binary tag string, - "categorical": 2D numpy "input" array of encoded tags while running single. Supports multi-label outputs. - : images identical to the input images (mostly used for working with autoencoders), - "multi_output": list with values ​​of different columns, None, no target is returned (the constructor only returns aggregated image data, which is useful to use in model.predict()).
Subset: It means subset. I set it to 0.1 as we were asked to set it to 0.1 if the subset of data (“training” or “validation”) was set in the validation split.
Seed: I set the optional random seeding to 0 for the shuffle operation and transformations.

I used the labels parameter to output the labels of the classes in the Train dataset.
There are 206 separated validation data that I have separated with 1881 training data validation_split. There are 6 classes in our dataset. These; cardboard, glass, metal, paper, plastic and trash.

My model consists of an artificial neural network consisting of fully dependent layers. It consists of 5 Dense layers and 1 Flatten. Since it is image data, we need to pass it through Flatten, which is the flattening process. I made the model input as the input shape (100,100,3) as I sized it in target size. I made the dense layers as 32,16,64,32 as input and 6, which is our category number as the number of classes I will output. Since my activation function is single label multiple classes classification, I set it as 'softmax'. I have set my Loss function as categorical_crossentropy. I set the optimization function as the guy with the best results and my metrics as accuracy. After specifying my functions, I compile the model.

 You can experience the created repo and report your problems. thanks.
