import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prepare_dataset import create_dataset
from models.sign_language_model import SignLanguageModel
import tensorflow as tf

dataset, class_names = create_dataset('data')
num_classes = len(class_names)

train_size =  int(0.8*len(list(dataset)))
train_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
val_dataset = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)

model = SignLanguageModel(num_classes)
model.summary()
model_callback = tf.keras.callbacks.ModelCheckpoint(filepath = 'models/sign_language_model_{epoch:02d}_{val_loss:.2f}.h5',
                                                    monitor = 'val_loss',
                                                    save_best_only=True,
                                                    mode='min',
                                                    verbose=1)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, verbose=1, callbacks=[model_callback])
# model.save('models/sign_language_model3.keras')

# Saving the parameter for visulaization
with open("models/training_data.bat", 'wb') as f:
    pickle.dump(history, f)