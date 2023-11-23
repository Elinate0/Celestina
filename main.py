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

# Sanatçıları resim sayısına göre sıralama
artists = artists.sort_values(by=['paintings'], ascending=False)

# 200'den fazla tabloya sahip sanatçıların bulunduğu bir veri seti veya çerçevesi oluşturma
artists_top = artists[artists['paintings'] >= 200].reset_index()
artists_top = artists_top[['name', 'paintings']]

#artists_top['class_weight'] = max(artists_top.paintings)/artists_top.paintings tekrar bakılması gerek
artists_top['class_weight'] = artists_top.paintings.sum() / (artists_top.shape[0] * artists_top.paintings)
artists_top
#çıktı https://gcdnb.pbrd.co/images/uIutkEMObxMq.png

# Sınıf ağırlıklarını ayarlamak gerek - yeterince temsil edilmeyen sınıflara daha yüksek ağırlıklar atamalıyız
class_weights = artists_top['class_weight'].to_dict()
class_weights
#Çıktısı
#{0: 0.44563076604125634,
# 1: 0.5567210567210568,
# 2: 0.8902464278318493,
# 3: 1.1631493506493507,
# 4: 1.1915188470066518,
# 5: 1.2566501023092662,
# 6: 1.3430178069353327,
# 7: 1.491672449687717,
# 8: 1.5089505089505089,
# 9: 1.532620320855615,
# 10: 1.6352225180677062}

# 'Albrecht_Dürer'i tanımakta bazı sorunlar var (nedenini bilmiyorum, bakmak gerek şimdilik bu şekilde fix)
updated_name = "Albrecht_Dürer".replace("_", " ")
artists_top.iloc[4, 0] = updated_name

# Explore images of top artists
images_dir = '../input/images/images'
artists_dirs = os.listdir(images_dir)
artists_top_name = artists_top['name'].str.replace(' ', '_').values

# Tüm dizinlerin mevcut olup olmadığına bakılması gerek
for name in artists_top_name:
    if os.path.exists(os.path.join(images_dir, name)):
        print("Found -->", os.path.join(images_dir, name))
    else:
        print("Did not find -->", os.path.join(images_dir, name))
#Çıktısı
#Found --> ../input/images/images/Vincent_van_Gogh
#Found --> ../input/images/images/Edgar_Degas
#Found --> ../input/images/images/Pablo_Picasso
#Found --> ../input/images/images/Pierre-Auguste_Renoir
#Found --> ../input/images/images/Albrecht_Dürer
#Found --> ../input/images/images/Paul_Gauguin
#Found --> ../input/images/images/Francisco_Goya
#Found --> ../input/images/images/Rembrandt
#Found --> ../input/images/images/Alfred_Sisley
#Found --> ../input/images/images/Titian
#Found --> ../input/images/images/Marc_Chagall

# Random artistlerin resimlerini getirmeyi deneyelim
n = 5
fig, axes = plt.subplots(1, n, figsize=(20,10))

for i in range(n):
    random_artist = random.choice(artists_top_name)
    random_image = random.choice(os.listdir(os.path.join(images_dir, random_artist)))
    random_image_file = os.path.join(images_dir, random_artist, random_image)
    image = plt.imread(random_image_file)
    axes[i].imshow(image)
    axes[i].set_title("Artist: " + random_artist.replace('_', ' '))
    axes[i].axis('off')

plt.show()
#Çıktısı https://gcdnb.pbrd.co/images/IQQBhRzo175k.png

# Verilerin düzenlenmesi
batch_size = 16
train_input_shape = (224, 224, 3)
n_classes = artists_top.shape[0]

#Buradaki bazı bilgileri kütüphaneden direkt aldım fakat çalışmayanlar olduğu için onları yorum satırına çektim
train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale=1./255.,
                                   #rotation_range=45,
                                   #width_shift_range=0.5,
                                   #height_shift_range=0.5,
                                   shear_range=5,
                                   #zoom_range=0.7,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                  )

train_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                    class_mode='categorical',
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=batch_size,
                                                    subset="training",
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist()
                                                   )

valid_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                    class_mode='categorical',
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=batch_size,
                                                    subset="validation",
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist()
                                                   )

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
print("Total number of batches =", STEP_SIZE_TRAIN, "and", STEP_SIZE_VALID)
