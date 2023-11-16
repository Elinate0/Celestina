#Veri setinde 50 sanatçının tablosu bulunmaktadır. Ancak burada sadece 11 sanatçının 200'den fazla tablosu mevcut.
#Hesaplamayı azaltmak ve daha iyi eğitim vermek için yalnızca bu 11 sanatçının resimlerini kullanmaya karar verdim.
#Bu dengesiz bir veri kümesi olduğundan (Van Gogh'un 877 tablosu varken Marc Chagall'ın yalnızca 239 tablosu var), class_weight önemlidir

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm, tqdm_notebook
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from numpy.random import seed
seed(1) #test amaçlı
from tensorflow import set_random_seed
set_random_seed(1) #test amaçlı

print(os.listdir("../input"))
#Çıktısı ['images', 'artists.csv', 'resized'] olacaktır

artists = pd.read_csv('../input/artists.csv')
artists.shape
#Çıktısı (50, 8) olacaktır.