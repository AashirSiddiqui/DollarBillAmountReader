## Dollar Bill Amount Reader

This model uses the ImageNet network from https://github.com/dusty-nv/jetson-inference to create an image classification network that can classify one dollar and five dollar bills' fronts and backs.

### Usage

Make sure to install https://github.com/dusty-nv/jetson-inference in the same directory as this repository before beginning.
Example directories for both jetson-inference and dollar-bill-amount-reader:
/home/machine-learning/jetson-inference
/home/machine-learning/dollar-bill-amount-reader

#### Opening the docker container

For training the model or predicting with the model, the docker container will need to be opened.
``cd jetson-inference``
``./docker/run.sh --volume /PATH/TO/dollar-bill-amount-reader:/jetson-inference/dollar-bill-amount-reader``
This will open the docker container and link the `/PATH/TO/dollar-bill-amount-reader` folder on your hard drive to the `/jetson-inference/dollar-bill-amount-reader` folder in the docker container.

#### Re-training the model

To train the model, the dusty-nv/jetson-inference docker container should be used.
Before starting the docker, to prevent the training crashing from overusing memory, we should enable ``/proc/sys/vm/overcommit_memory``. Open a terminal (which is not in a docker container) and type in the following command:
`echo 1 | sudo tee /proc/sys/vm/overcommit_memory`

Then, open the docker as shown in the above **Opening the docker container** section.

To train the model, execute the `train.py` file for classification networks from  https://github.com/dusty-nv/jetson-inference inside the docker terminal.
```
cd /jetson-inference/python/training/classification;
python3 train.py --model-dir=/jetson-inference/dollar-bill-amount-reader/models/bill-reader /jetson-inference/dollar-bill-amount-reader/data/bill-reader --epochs=EpochAmount;
```

Once that command finishes, two model files will be generated inside `model-dir`: `checkpoint.pth.tar` and `model_best.pth.tar`. To predict with these models, they will have to be exported to ONNX format. The following command, executed in the same directory as `train.py` was executed, will convert both of them to ONNX format using the https://github.com/dusty-nv/jetson-inference `onnx_export.py` file.

```
python3 onnx_export.py --input /jetson-inference/dollar-bill-amount-reader/models/bill-reader/model_best.pth.tar --output /jetson-inference/dollar-bill-amount-reader/models/bill-reader/model_best.onnx; python3 onnx_export.py --input /jetson-inference/dollar-bill-amount-reader/models/bill-reader/checkpoint.pth.tar --output /jetson-inference/dollar-bill-amount-reader/models/bill
-reader/checkpoint.onnx;
```

#### Predicting with the model (one image file)

First, open the docker as shown in the above **Opening the docker container** section.

Then, once inside the docker, navigate to `/jetson-inference/dollar-bill-amount-reader` and execute `predict-bill.py` referencing the model and test image. The prediction will be printed.

```
cd /jetson-inference/dollar-bill-amount-reader;
python3 predict-bill.py --model=./models/bill-reader/ONNX_MODEL ./data/bill-reader/IMAGE_FILE;
```
#### Predicting with the model (camera stream)

First, open the docker as shown in the above **Opening the docker container** section.

Once inside the docker, navigate to `/jetson-inference/dollar-bill-amount-reader` and execute `camerastream.py` adding your webcam id (e.g: `/dev/video0`) as an argument.

```
cd /jetson-inference/dollar-bill-amount-reader
python3 camerastream.py /dev/video#

```
