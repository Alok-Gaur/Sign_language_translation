import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LSTM
from tensorflow.keras import Model

class SignLanguageModel(Model):
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()

        self.dense_input = Dense(128, activation='relu', name="Dense_Input_Layer")
        self.dense1 = Dense(128, activation='relu', name='dense1_relu')
        self.dense2 = Dense(45, activation='tanh', name='dense2_tanh')
        self.dense3 = Dense(45, activation='sigmoid', name="dense3")

        self.lstm = LSTM(64, return_sequences=False)

        self.post_dense = Dense(64, activation='relu', name='post_dense')
        self.output_layer = Dense(num_classes, activation='softmax', name='output')
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
        self.batch_norm3 = BatchNormalization()

    def call(self, inputs, training=False):
        x1 = self.dense_input(inputs)
        x1 = self.batch_norm1(x1, training=training)
        x1 = self.dense1(x1)
        x2 = self.dense2(x1)
        x3 = self.dense3(x1)
        
        x = tf.add(x2,x3, name="add")

        x = self.batch_norm2(x, training=training)
        x = tf.expand_dims(x, axis=1)
        x = self.lstm(x)

        x = self.post_dense(x)
        x = self.batch_norm3(x, training=training)
        return self.output_layer(x)

