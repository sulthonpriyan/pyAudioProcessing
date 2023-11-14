import os
import gc
os.environ['PA_ALSA_DEBUG'] = '0'
from pydub.playback import play
from audio import Audio
from microphone import Microphone
import spafe
from spafe.utils.vis import show_features
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
import numpy as np
from scipy.io import wavfile
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import tensorflow_io as tfio
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import cv2 as cv
import tensorflow 
from tensorflow import keras
import matplotlib.pyplot as plt
# import tensorflow_io as tfio

# ArduinoSerial = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
# time.sleep(1)

webcam = Microphone(device_index=1)
audio = webcam.capture_audio(access_time=1.5)
room_voice = Audio(audio)
#audio = room_voice.noise_reduce()
room_voice.save_chunks(
        audio=audio, 
        export_dir="E:/Perkuliahan Duniawi/Skripsi/room-voice-command-dataset-code/model/Audio", 
        file_name=f"temp" 
    )

rate, data = wavfile.read("E:/Perkuliahan Duniawi/Skripsi/room-voice-command-dataset-code/model/Audio/temp.wav")
gfccs  = gfcc(data,
                  fs=16000,
                  pre_emph=1,
                  pre_emph_coeff=0.97,
                  window=SlidingWindow(0.03, 0.015, "hamming"),
                  nfilts=128,
                  nfft=2048,
                  low_freq=0,
                  high_freq=8000,
                  normalize="mvn")
plt.figure(figsize=(14,4))
plt.imshow(gfccs.T, origin="lower", aspect="auto", cmap="jet", interpolation="nearest")
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
plt.savefig('E:/Perkuliahan Duniawi/Skripsi/room-voice-command-dataset-code/model/Audio/temp.png')




class_names = ['Maju','Mundur','Kanan','Kiri','Berhenti']

model = keras.models.load_model('E:/Perkuliahan Duniawi/Skripsi/room-voice-command-dataset-code/model/Model_ResNet50New')
img = tf.keras.preprocessing.image.load_img('E:/Perkuliahan Duniawi/Skripsi/room-voice-command-dataset-code/model/Audio/temp.png', target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.array([img_array])
predictions = model.predict(img_array)
print(predictions)
class_id = np.argmax(predictions, axis = 1)
print(class_id)
result = class_names[class_id.item()]
print(result)
