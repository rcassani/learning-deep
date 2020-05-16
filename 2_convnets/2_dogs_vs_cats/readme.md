# CNN to classify Dogs vs Cats dataset

This is CNN classifier trained for the Dogs vs Cats InclassKaggle challenge for image classification.
The challege was part of the assignment #1 for the course [IFT 6135 Representation Learning
Winter 2019](https://sites.google.com/mila.quebec/ift6135), a Deep Learning Course offered by the Université de Montréal.

Data available in:
[https://www.kaggle.com/c/ift6135h19/overview](https://www.kaggle.com/c/ift6135h19/overview)

## Architecture
The model that was used is described in [`model_zoo.py`](model_zoo.py)) with the name `large_cnn_2` The input for is a 3-channel image of 64 by 64 pixels, and the output is 2 classes (dogs and cats).

<p align="center">
<img src="https://user-images.githubusercontent.com/8238803/82124583-f96b2700-976d-11ea-962c-4d3fe7ff3424.png", width=500>
<br>
<code>mlp_simple</code> model, input is 3<span>&#215;</span>64<span>&#215;</span>64
</p>

in the dataset there are 9999 samples for each class. This dataset was divided in 70%, 20% and 10% for train, validation and test datasets respectively. Additionally, the train and validation datasets were augmented by concatenating them with a its own transformed copy (random horizontal flip and random rotation, see `train_dnn.py` for details.)


## Training the model
The model can be trained from the terminal with the command:  
```bash
python -i train_dnn.py --epochs=100 --ngpus=1 --lr=0.5 --l2=1 --model=large_cnn_2
```
alternatively, `the final_state.pt` file can be loaded as:
```bash
python -i train_dnn.py --epochs=100 --ngpus=1 --model=large_cnn_2 --final-state
```

## Learning curves and accuracy
The model was trained with a learning rate of 0.5, L2 of 1, SGD as optimizer and cross cross entropy as loss function. Below the plots for loss and accuracy vs epochs.

|   |   |
|---|---|
| <img src="https://user-images.githubusercontent.com/8238803/82125825-a5b10b80-9776-11ea-890d-8dab2f1c6bf1.png" style="width: 400px;"/> | <img src="https://user-images.githubusercontent.com/8238803/82125829-a8136580-9776-11ea-949c-7f31b7a009f6.png" style="width: 400px;"> |
| loss | accuracy |

The accuracy for the models was 96.42% on the test set.

## Examples of classification
These are some examples from the test set, [T]rue and [P]redicted labels are indicated. These are predicted with the `large_cnn_2` model.

<p align="center">
<img src="https://user-images.githubusercontent.com/8238803/82125833-a9dd2900-9776-11ea-84ad-1e0976af0620.png" style="width: 100%;">
Classification results with `large_cnn_2`model
</p>

Finally, I used pictures of Poupune (cat) and Spock (dog) as input to the CNN, and these are the results.
<p align="center">
<img src="https://user-images.githubusercontent.com/8238803/82125839-b1043700-9776-11ea-8b4e-4b2b67884bce.png" style="width: 100%;">
Classification results with `large_cnn_2`model
</p>
