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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
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
conv_layer_0 = Conv1D(filters=64,kernel_size=6,padding='same',activation='relu')(layer) #1D слой свёртки
conv_layer = Conv1D(filters=32,kernel_size=3,padding='same',activation='relu')(conv_layer_0)
concat_layer = MaxPooling1D(pool_size=2)(conv_layer) #Max операция объединения для 1D временных данных для уменьшения выборки представления
mem_layer = LSTM(150)(concat_layer) #слой LST-памяти

output = Dense(top, activation='softmax')(mem_layer)
model = Model(inputs=input, outputs=output) #Модель, группирующая слои в объект с функциями обучения/вывода

#Обучение и тестирование
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
print(model.summary())
X_seq = np.array(X_seq)
X_seq = np.array(list(X_seq))
b = np.random.rand(len(df)) < 0.8

X_seq_train = X_seq[b]
a_train = a[b]

X_seq_test = X_seq[~b]
a_test = a[~b]

early_stop = EarlyStopping(monitor='val_loss',patience=5)

#Сохранение результатов по разным параметрам модели
model_save = ModelCheckpoint(
    '/working/best_model.h5', monitor='val_acc', verbose=1, save_best_only=True,
    save_weights_only=False, mode='auto')

#история
h = model.fit(x=X_seq_train, y=a_train, batch_size=128, epochs=15, verbose=1, validation_split=0.2,callbacks=[model_save])

best_model = tf.keras.models.load_model('/working/best_model.h5')

a_test_0 = best_model.predict(X_seq_test) #предсказание
confusion_Matrix = confusion_matrix(np.argmax(a_test,axis=1), np.argmax(a_test_0,axis=1))

# Построение нормализованной матрицы
confusion_Matrix = confusion_Matrix.astype('float') / confusion_Matrix.sum(axis=1)[:, np.newaxis]
np.set_printoptions(precision=2)
plt.figure(figsize=(10,10))
plt.imshow(confusion_Matrix, interpolation='nearest', cmap=plt.cm.Reds)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(top)
plt.xticks(ticks, sort, rotation=90)
plt.yticks(ticks, sort)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(confusion_Matrix)