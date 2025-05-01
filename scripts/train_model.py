import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prepare_dataset import create_dataset
from models.sign_language_model import SignLanguageModel2
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

dataset, class_names = create_dataset('data')
num_classes = len(class_names)

train_size =  int(0.8*len(list(dataset)))
train_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
val_dataset = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)

model = SignLanguageModel2(num_classes)

#Checkpoint to save the best weights
model_callback = tf.keras.callbacks.ModelCheckpoint(filepath = 'saved_models/model_2_weights1/sign_language_model_{epoch:02d}_{val_loss:.2f}.keras',
                                                    monitor = 'val_loss',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    mode='min',
                                                    verbose=1)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

history = model.fit(train_dataset, validation_data=val_dataset, epochs=70, verbose=1, callbacks=[model_callback])
model.save('saved_models/sign_language_model5.keras')


# Saving the parameter for visulaization
with open("results/training_data2.pkl", 'wb') as f:
    pickle.dump(history.history, f)