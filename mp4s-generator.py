from gtts import gTTS
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("targetdir", type=str, help="the directory to generate the mp4s inside")
opt = parser.parse_args()

## parser invalid arguments error handling
if not os.path.isdir(opt.targetdir):
    print("Error: invalid target directory- not a directory or directory not found")
    exit()

toMake = [
    ["1dollar_F.mp4","One dollar bill, front."],
    ["1dollar_B.mp4","One dollar bill, back."],
    ["5dollar_B.mp4","Five dollar bill, back."],
    ["5dollar_F.mp4","Five dollar bill, back."]
          ]

tts = gTTS('hello')
tts.save('./mp4s/hello.mp3')

tts = gTTS('hello')
tts.save('hello.mp3')