# MLP to classify MNIST dataset

## Architecture
The used architecture or model is described in [`model_zoo.py`](model_zoo.py)) with the name `mlp_simple`. It is a n MLP with layer sizes: [784, 25, 10, 10], this is to say, 2 hidden layers. ReLU was used as activation function.

<p align="center">
<img src="https://user-images.githubusercontent.com/8238803/82124583-f96b2700-976d-11ea-962c-4d3fe7ff3424.png" width=500>
`mlp_simple` model, input is 1<span>&#215;</span>784
</p>

## Training the MLP
The model can be trained from the terminal with the command:  
```bash
python -i train_dnn.py --epochs=50 --ngpus=1 --lr=0.5 --l2=1 --model=mlp_simple
```
alternatively, `the final_state.pt` file can be loaded as:
```bash
python -i train_dnn.py --ngpus=1 --model=mlp_simple --final-state
```

## Learning and Accuracy curves
The models was trained with a learning rate of 0.5, L2 of 1, a maximum number of 50 epochs, SGD as optimizer and cross cross entropy as loss function. Below the plots for loss and accuracy vs epochs.

|   |   |
|---|---|
| <img src="https://user-images.githubusercontent.com/8238803/80164966-55e08980-85a8-11ea-84f0-62c81b3bbdd3.png" style="width: 400px;"/> | <img src="https://user-images.githubusercontent.com/8238803/80164976-5bd66a80-85a8-11ea-9c6b-a4e625924fce.png" style="width: 400px;"> |
| loss | accuracy |

## Examples of classification
These are some examples from the test set, [T]rue and [P]redicted labels are indicated.

<p align="center">
<img src="https://user-images.githubusercontent.com/8238803/80164991-6690ff80-85a8-11ea-9a75-1df8b8d5183f.png" style="width: 100%;"/>
Classification results with <code>mlp_simple</code> model
</p>

## Plotting some weights
Lastly, these are the plots for the weights between the input layer and the first hidden layer

<p align="center">
<img src="https://user-images.githubusercontent.com/8238803/80164987-642ea580-85a8-11ea-9e99-f9cc02f3db77.png" style="width: 100%;"/><br>
Weights <code>W1</code> in <code>mlp_simple</code> model
</p>
