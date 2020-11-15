# Automatic Multi Layer Perceptron training through Meta-Learning
This example code shows how to use a simple genetic algorithm to create a population of Multi Layer Perceptron (MLP)
networks and improve their accuracy on a specific dataset by using validation accuracy as fitness function.

Each MLP's hyper-parameters are different and they can have multiple choices. For example, all choices for the number of
neurons in a dense layer is represented by a list of integers. The same procedure is used for the number of layers we
 want to be used in our networks.
Activations and optimizers are represented as strings. We're currently not dealing with learning rates, but it could be
 easily integrated in the framework.

It'll take a while to train networks so take your time to relax.

On the easy MNIST dataset, it's straightforward to quickly find a network that reaches > 98% accuracy.

## Run in a conda environment
You can create a conda environment with the provided `requirements.yml` where all dependencies are listed.
```shell script
conda env create -f resources/environment.yml 
conda activate aiml
python src/main.py
``` 

## Run in a Docker container
You can build the Docker image by Dockerfile
```
docker build . -t meta-learning:0.0.1
docker run -it meta-learning:0.0.1
```

### Credits
The genetic optimizer code is inspired by the code from this blog post: https://lethain.com/genetic-algorithms-cool-name-damn-simple/

### Contributing
Have an optimization, idea, suggestion, bug report? Pull requests greatly appreciated!

### License
MIT
