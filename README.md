# LocalLearning
Exploring Krotov and Hopfield's local learning rules, possibly improving robustness of classifiers.

# Using CPU: Docker Container
```bash
sudo chown 1000:1000 LocalLearning/src
sudo chown 1000:1000 LocalLearning/notebooks
sudo chwon 1000:1000 LocalLearning/data
docker build -t pytorch-cpu-dev .
docker run -v $PWD:/pytorch-dev/LocalLearning -p 3108:3108 -it pytorch-cpu-dev
```
# Using GPU: Anaconda virtual environment
```bash
conda env create -f pytorch.yml
```
