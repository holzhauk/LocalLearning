# LocalLearning
Exploring Krotov and Hopfield's local learning rules, possibly improving robustness of classifiers.
# Using CPU: Docker Container
'''bash
sudo chown 1000:1000 LocalLearning/src
sudo chown 1000:1000 LocalLearning/notebooks
docker build -t pytorch-cpu-dev
docker run -v $PWD/src:/pytorch-dev/src -v $PWD/notebooks/pytorch-dev/notebooks -it pytorch-cpu-dev
'''
