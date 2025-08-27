# Pumpkin Seed Classification using a Simple Neural Network

## Project Description

This project demonstrates the classification of pumpkin seeds into two classes ('Çerçevelik' and 'Ürgüp Sivrisi') using a simple feedforward neural network implemented with PyTorch. The project involves data loading, preprocessing, model definition, training, and evaluation.

## Dataset

The dataset used in this project is the "Pumpkin Seeds Dataset". It contains various features of pumpkin seeds, such as Area, Perimeter, Major Axis Length, Minor Axis Length, etc., and the corresponding class label.

The dataset was loaded from the Excel file `./content/pumpkin_seeds_dataset/Pumpkin_Seeds_Dataset.xlsx`.

## Data Preprocessing

1.  **Loading Data:** The dataset was loaded into a pandas DataFrame.
2.  **Encoding Class Labels:** The 'Class' column, which contains categorical labels ('Çerçevelik' and 'Ürgüp Sivrisi'), was converted into numerical labels (0 and 1) using a mapping dictionary.
3.  **Splitting Data:** The dataset was split into training and testing sets using `train_test_split` from `sklearn.model_selection`.
4.  **Converting to Tensors:** The features (X) and labels (y) were converted into PyTorch Tensors with appropriate data types (FloatTensor for features, LongTensor for labels).

## Model Architecture

A simple feedforward neural network was used for classification. The model consists of:

-   An input layer with 12 features (corresponding to the 12 features in the dataset).
-   A first hidden layer with 9 neurons and a ReLU activation function.
-   A second hidden layer with 13 neurons and a ReLU activation function.
-   An output layer with 2 neurons (corresponding to the two classes) and no activation function (as `CrossEntropyLoss` is used).

The model is defined in the `Model` class, inheriting from `torch.nn.Module`.

## Training

-   **Criterion:** `nn.CrossEntropyLoss()` was used as the loss function.
-   **Optimizer:** `torch.optim.Adam` was used as the optimizer with a learning rate of 0.00001.
-   **Epochs:** The model was trained for 20 epochs.
-   **Batch Size:** A batch size of 32 was used for training and evaluation.
-   **Data Loaders:** `DataLoader` was used to efficiently load data in batches during training and evaluation.

The training process included iterating through the training data in batches, calculating the loss, performing backpropagation, and updating the model's weights. The training and test loss were monitored and printed for each epoch.

## Evaluation

The model was evaluated on the test dataset after training.

-   The loss on the test set was calculated using `nn.CrossEntropyLoss()`.
-   The accuracy of the model was calculated by comparing the predicted class (the class with the highest output score) with the actual class for each sample in the test set.

## Results

-   The training and test loss decreased significantly over the epochs, indicating that the model was learning.
-   The final test loss was approximately 0.4133.
-   The accuracy of the model on the test set was calculated and printed, showing the number of correct predictions out of the total number of test samples and the corresponding percentage.


