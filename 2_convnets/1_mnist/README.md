# CNN to classify MNIST dataset

## Architectures
Two models were used, these described in [`model_zoo.py`](model_zoo.py)) with the name `small_cnn` and `medium_cnn`. The input for both models is a 1-channel image of 28 by 28 pixels, and the output is 10 classes (digits).

|   |   |
|---|---|
| <img src="https://user-images.githubusercontent.com/8238803/82125015-00dfff80-9771-11ea-950b-ca8c86475bda.png" style="width: 400px;"> | <img src="https://user-images.githubusercontent.com/8238803/82125017-02a9c300-9771-11ea-8f8d-ace6d202c653.png" style="width: 400px;"> |
| `small_cnn` | `medium_cnn` |

## Training the models
The model can be trained from the terminal with the command:  
```bash
python -i train_dnn.py --epochs=50 --ngpus=1 --lr=0.5 --l2=1 --model=small_cnn
```
alternatively, `the final_state.pt` file can be loaded as:
```bash
python -i train_dnn.py --ngpus=1 --model=small_cnn --final-state
```

## Learning curves and accuracy
The models was trained with a learning rate of 0.5, L2 of 1, a maximum number of 50 epochs, SGD as optimizer and cross cross entropy as loss function. Below the plots for loss vs epochs.

|   |   |
|---|---|
| <img src="https://user-images.githubusercontent.com/8238803/81992475-83ea4400-9611-11ea-8ab8-3ca306cbbea3.png" style="width: 400px;"/> | <img src="https://user-images.githubusercontent.com/8238803/81992478-86e53480-9611-11ea-889e-2749cf3e83df.png" style="width: 400px;"/> |
| `small_cnn` | `medium_cnn` |

The accuracy for both models in the test set was: 98.50% and 98.33%, for `small_cnn` and `medium_cnn` respectively. Although the `small_cnn` model took 56 epochs to get there, while the `medium_cnn` model just took 29 epochs.

## Examples of classification
These are some examples from the test set, [T]rue and [P]redicted labels are indicated. These are predicted with the `medium_cnn` model.

<p align="center">
<img src="https://user-images.githubusercontent.com/8238803/81992485-8b115200-9611-11ea-95aa-ed884048295d.png" style="width: 100%;"/>
Classification results with `medium_cnn`model
</p>

## Plotting the kernels
Lastly, in both models, the first layer is a convolutional layer of 10 kernels and size 3 by 3 with padding 1.

<p align="center">
<img src="https://user-images.githubusercontent.com/8238803/81992486-8d73ac00-9611-11ea-9394-abda5b717b50.png" style="width: 100%;"/>
10 kernels for the first layer in the `medium_cnn`model
</p>

From the kernels it is possible to see some of learnt properties. For example kernel `9` is used to detect horizontal edges with white on top and black on the bottom. By using these convolutional kernels into the input image, its feature maps are obtained. Below 4 examples of input images and their feature maps. The original image is in the bottom right plot for each figure.

|   |   |
|---|---|
| <img src="https://user-images.githubusercontent.com/8238803/81992494-92d0f680-9611-11ea-8103-38003e5d5b5e.png" style="width: 400px;"> | <img src="https://user-images.githubusercontent.com/8238803/81992498-96647d80-9611-11ea-92fa-772e0caa2206.png" style="width: 400px;"> |

 <p align="center">
<img src="https://user-images.githubusercontent.com/8238803/81992523-a54b3000-9611-11ea-9bdd-35c13ba78b1d.png" width=400><br>  
10 kernels for the first layer in the `medium_cnn`model
</p>
