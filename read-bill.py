#!/usr/bin/python3

import jetson_inference
import jetson.utils

import argparse

parser = argparse.ArgumentParser() # argparse can be used to code simple cli applications (e.g: python3 my-recognition.py --help, python3 my-recognition.py --filename test.jpg --network googlenet)
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--model", type=str, default="models/bill-reader/model_best.onnx", help="model to use, by default: models/bill-reader/model_best.onnx")
parser.add_argument("--labels", type=str, default="models/bill-reader/labels.txt", help="labels to use, by default: models/bill-reader/labels.txt")
opt = parser.parse_args() # opt = Namespace for cli operation (e.g: Namespace(filename='filename', network='googlenet'))
# print(opt)

image = jetson.utils.loadImage(opt.filename)
net = jetson_inference.imageNet(model=opt.model, labels=opt.labels, input_blob="input_0", output_blob="output_0")

class_idx, confidence = net.Classify(image) # class idx = classification #, confidence = how sure network is in decimal value (0 = not sure at all, 1 = completely sure)
class_desc = net.GetClassDesc(class_idx) # class names

print("image is recognized with "+str(confidence * 100) +"% confidence as: "+str(class_idx)+" ("+str(class_desc)+").")