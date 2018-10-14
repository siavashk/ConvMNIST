# ConvMNIST
MNIST Classification Using Convolutional Neural Network.

MIT License.

## Introduction
Note: Skip this section if you are only interested in running this code.

This is a Tensorflow implementation of convolutional neural network (CNN) for classification of handwritten images. This dataset is known as [MNIST](http://yann.lecun.com/exdb/mnist/) in the machine learning community.

This repository contains code for training the CNN, henceforth referred to as the model, through `scripts/train.py`. It also contains code for classifying images supplied either through the command line (`scripts/inference_cmd.py`) or by making a POST request to a RESTful inference service (`scripts/inference_server.py`).

The `core` folder contains layer definitions (`core/layers.py`) and dataset providers (`core/mnistbatch.py`). The full CNN is defined in `core/networks.py`.

The frozen model, which is the result of training, is defined under `moder/frozenmodel.pb`. I have also provided sample images for testing the trained model under `images/*.jpg`.

## Requirements
* [Tensorflow](https://www.tensorflow.org/install/)
* [tfserve](https://github.com/iitzco/tfserve)
* [Pillow](https://pillow.readthedocs.io/en/5.3.x/)
* Anaconda (optional)

## GPU Installation (optional)
If you would like to perform training/inference on the GPU, you will need to install Nvidia drivers for your graphics card. You can download the most recent drivers [here](https://www.nvidia.com/download/index.aspx).

You will also need to install CUDA. Tensorflow 1.11 requires at least CUDA 9.0, which can be downloaded [here](https://developer.nvidia.com/cuda-toolkit-archive).

Finally, you will also need to download CUDNN. Follow the instructions [here](https://developer.nvidia.com/cudnn).

## Anaconda Installation
Create an anaconda virtual environment such as:
```bash
conda create -n pytf_36 python=3.6
```
Activate the environment:
```bash
conda activate pytf_36
```
Depending on if you installed GPU requirements, install `Tensorflow`, `tfserve` and `Pillow`:
```bash
pip install -r requirements_cpu.txt
```
or
```bash
pip install -r requirements_gpu.txt
```

## Training
Before starting to train, add the `core` folder to your `PYTHONPATH`. This can be done by sourcing `env.sh` from any location on your harddrive.
```bash
source env.sh
```
You can start training using default arguments simply by running:
```bash
python scripts/train.py
```
Look at `script/train.py` for adjusting command-line arguments that adjust training hyper parameters. After training, I get about 97% and 96% accuracy on training and testing batches.

## Inference Command Line Tool
Simply run:
```bash
python scripts/inference_cmd.py
```
to try out the command line tool. See the argument parser for supplying different images to command line tool.

## Inference Server
First, start the server by running:
```bash
python scripts/inference_server.py
```
This starts the inference server on local host and accepts POST requests on port 5000. The host and port can be changed through command line arguments at startup.

You can curl or use any tool to submit the POST request. I prefer [Postman](https://www.getpostman.com/), but you are free to use whatever you like.

### Using Postman to Submit a POST Request
Open the Postman application and select a POST request as shown in Figure 1. <br/>
![Post request](/assets/post.png "Figure 1: Select the Post request from the drop down menu.")<br/>
Set the local host and listening port as shown in Figure 2. <br/>
![Post request](/assets/localhost.png "Figure 2: Set the localhost and listening port.")<br/>
Finally, set the request `body` as `binary` and choose a `jpg` file as the payload. Submit the post request by clicking the `Submit` button as shown in Figure 3. <br/>
![Post request](/assets/body.png "Figure 3: Submit the request.")<br/>
You should receive a JSON response containing `label` and `probability` fields as shown in Figure 4. <br/>
![Post request](/assets/response.png "Figure 4: JSON response containing label and probability.")<br/>
