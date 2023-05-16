#Project final version
import numpy as np
import pandas as pd
from collections import Counter
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense,Dot,Conv1D,MaxPooling1D,Activation,Dropout,LSTM,Flatten,GlobalMaxPooling1D,Input
import matplotlib.pyplot as plt
import seaborn as sns
#считываем данные (из pdb)
seq = pd.read_csv('pdb_data_seq.csv')
data = pd.read_csv('pdb_data_no_dups.csv')
df = data.merge(seq,how='inner',on='structureId').drop_duplicates(['structureId'])
#предобработка
print(df.isnull().sum())
df = df.dropna()
df = df[df['macromoleculeType_x'] == 'Protein']
df = df.reset_index()
df = df.drop(['index'],axis=1)
print(df.residueCount_x.describe())
df = df.loc[df.residueCount_x<1200]
df['merge_col'] = list(map(lambda i : str(i['sequence']) + ' ' + str(i['residueCount_y'])+" "+str(i['phValue'])+" "+str(i['densityPercentSol'])+" "+str(i['macromoleculeType_y']) + " " + str(i['residueCount_x']) + " " + str(i['resolution']) + " " + str(i['crystallizationTempK']) + " " + str(i['chainId']) ,df.to_dict(orient="records") ))
#количество в каждом классе
k = Counter(df.classification)
top = 10
sort = k.most_common()[:top]
classes = [a[0] for a in sort]
count = [a[1] for a in sort]
print("минимум " + str(count[-1]) + " образцов в каждом классе")
#добавляем в датафрейм
df = df[[a in classes for a in df.classification]]
seqs = df['merge_col']
length = [len(s) for s in seqs]
print(df.classification.value_counts())

lab = LabelBinarizer()
a = lab.fit_transform(df.classification)
X=df.drop('classification',axis=1)
df.iloc[0]['merge_col']
max_len = max(length)
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(seqs)
X_seq = tokenizer.texts_to_sequences(seqs)
X_seq = tf.keras.utils.pad_sequences(X_seq, maxlen=max_len) #проверяем что риды одной длины
print(df.info())

#итоговая модель
input = Input(shape=(max_len,)) #последовательность

layer = Embedding(len(tokenizer.word_index)+1, 8)(input) #встраиваемый слой
CNN_Layer_pre = Conv1D(filters=64,kernel_size=6,padding='same',activation='relu')(layer)
CNN_layer = Conv1D(filters=32,kernel_size=3,padding='same',activation='relu')(CNN_Layer_pre)
MaxPooling_layer = MaxPooling1D(pool_size=2)(CNN_layer)
LSTM_Layer = LSTM(150)(MaxPooling_layer)

output = Dense(top, activation='softmax')(LSTM_Layer)
model = Model(inputs=input, outputs=output)