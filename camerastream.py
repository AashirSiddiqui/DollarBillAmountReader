#!/usr/bin/env python3
#
# This python file was derived from https://github.com/dusty-nv/jetson-inference/blob/master/python/examples/imagenet.py.
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse
from time import sleep

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("--model", type=str, default="models/bill-reader/model_best.onnx", help="pre-trained model to load (see below for options)")
parser.add_argument("--labels", type=str, default="models/bill-reader/labels.txt", help="class labels of models to use")
parser.add_argument("--topK", type=int, default=1, help="show the topK number of class predictions (default: 1)")
parser.add_argument("--delaysec", type=int, default=3, help="the # of seconds in between each camera capture and prediction (default: 3)")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


# load the recognition network
net = imageNet(model=args.model, labels=args.labels, input_blob="input_0", output_blob="output_0")

# note: to hard-code the paths to load a model, the following API can be used:
#
# net = imageNet(model="model/resnet18.onnx", labels="model/labels.txt", 
#                 input_blob="input_0", output_blob="output_0")

# create video sources & outputs
inputVS = videoSource(args.input, argv=sys.argv)
# output = videoOutput(args.output, argv=sys.argv)
font = cudaFont()

# process frames until EOS or the user exits
while True:
    sleep(args.delaysec)

    img = inputVS.Capture()

    if img is None: # timeout
        continue  

    # classify the image and get the topK predictions
    # if you only want the top class, you can simply run:
    #   class_id, confidence = net.Classify(img)
    predictions = net.Classify(img, topK=args.topK)

    # draw predicted class labels
    for n, (classID, confidence) in enumerate(predictions):
        classLabel = net.GetClassLabel(classID)
        confidence *= 100.0

        print(f"imagenet:  {confidence:05.2f}% class #{classID} ({classLabel})")

    # print out performance info
    net.PrintProfilerTimes()