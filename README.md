# Training Quantized Neural Networks
Take a look at _config/config.py_ to understand all the parameters
They can all be overrode from the command line if necessary (just check _train.sh_)


## How to use
_train.py_ is the main file and trains one network. This file also contains the **optimizer and loss selection**
_train.sh_ is our script that launches multiple times _train.py_. It needs 3 arguments to run and in that order:
- resnet size (nres in _config_)
- start learning rate (lr in _config_)
- which gpu (cuda in _config_)

## Layers
Contains the quantized layer operations and subsequent layers.
This code comes from B. Moons & the other repository from which he originally took the inspiration

## Models
_model_factory_ contains the layer types
_resnet_ and _vgg_ contain their respective networks

## Utils
_keras_utils_ contains a backwards compatible keras callback
_load_data_ is a helper that returns the different data-sets
_config_utils_ parses the configuration files
