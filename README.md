# EmotionRecognition

*A work in progress*

A PyTorch deep neural network for predicting images from the FER-2013 facial expression & emotion dataset.

Utilize the commands and optional parameters below.

## Commands

Terminal train command:

```
python main.py ../data/Xy.pickle --epochs 100 --lr 0.003 --lr_decay_gamma 0.98 --verbose --use_tensorboard --stdout_to_file
```
*Note: The output will be sent to a log file in the output directory*


Then, in a Jupyter notebook,
```
!pip install tensorboard
```
```
%reload_ext tensorboard
%tensorboard --logdir=".../EmotionRecognition/outputs/log0/" --reload_multifile True
```

## Argument Parsing

### Optional Parameters
- `--data`: Specify the data path.
- `--model_path`: Path from which to load model parameters. Must be specified if `--predict` flagged.
- `--batch_size`: Specify training batch size.
- `--lr`: Specify learning rate.
- `--lr_decay_gamma`: Specify `torch.optim.lr_scheduler.StepLR` gamma.
- `--epochs`: Specify maximum number of epochs.
- `--patience`: Specify early stopping patience.


### Flags
- `--predict`: If flagged, predict, otherwise train.
- `--stdout_to_file`: Send stdout go to a log file in the outputs folder. The only exception is that the tqdm progress bar will still appear in console.
- `--verbose`: Specify training verbosity.
- `--use_tensorboard`: Setups up TensorBoard, which can be used to visualize training live.
- `--test`: Test after training the model.
- `--load_in_batches`: Load the data in batches, rather than all at once. Recommended to not flag unless having out of memory problems, as training is much faster.
- `--visualize_images`: Visualize random images from the training set. Cannot flag `--stdout_to_file` or `--load_in_batches` when visualizing images.

## Pre-trained Model

Coming soon!

## Fun Features
- A `tqdm` training progress bar.
- A manual exit option using the `exit.txt` file. Save document with "1" to end the training, or "0" otherwise. Upon exiting, the program resets `exit.txt` to 0.
- Capability to train and visualize several runs with the same hyperparameters, as well as plot the 'average run'.