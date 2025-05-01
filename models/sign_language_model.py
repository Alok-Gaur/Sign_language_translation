import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, GRU, Dropout, LSTM
from tensorflow.keras import Model

class SignLanguageModel(Model):
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()
        l2 = tf.keras.regularizers.l2(0.001)
        self.dense_input = Dense(128, kernel_regularizer=l2, activation='relu', name="Dense_Input_Layer")
        self.dense1 = Dense(128, kernel_regularizer=l2, activation='relu', name='dense1_relu')
        self.dense2 = Dense(45, kernel_regularizer=l2, activation='tanh', name='dense2_tanh')
        self.dense3 = Dense(45, kernel_regularizer=l2, activation='tanh', name="dense3_tanh")

        self.lstm = LSTM(64, return_sequences=False)

        self.post_dense = Dense(64, kernel_regularizer=l2, activation='relu', name='post_dense')
        self.output_layer = Dense(num_classes, kernel_regularizer=l2, activation='softmax', name='output')
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
        self.batch_norm3 = BatchNormalization()

        self.dropout1 = Dropout(0.4)
        self.dropout2 = Dropout(0.3)
    
    def call(self, inputs, training=False):
        x1 = self.dense_input(inputs)
        x1 = self.batch_norm1(x1, training=training)
        x1 = self.dense1(x1)
        x1 = self.dropout1(x1)
        x2 = self.dense2(x1)
        x3 = self.dense3(x1)
        
        x = tf.concat([x2,x3], axis=1, name="concat")

        x = self.batch_norm2(x, training=training)
        x = tf.expand_dims(x, axis=1)
        x = self.lstm(x)
        x = self.dropout2(x)
        x = self.post_dense(x)
        x = self.batch_norm3(x, training=training)
        return self.output_layer(x)



class SignLanguageModel2(Model):
    def __init__(self, num_classes):
        super(SignLanguageModel2, self).__init__()
        l2 = tf.keras.regularizers.l2(0.001)
        self.dense_input = Dense(128, kernel_regularizer=l2, activation='relu', name="Dense_Input_Layer")
        self.dense1 = Dense(128, kernel_regularizer=l2, activation='relu', name='dense1_relu')
        self.dense2 = Dense(45, kernel_regularizer=l2, activation='tanh', name='dense2_tanh')
        self.dense3 = Dense(45, kernel_regularizer=l2, activation='tanh', name="dense3_tanh")

        self.gru = GRU(128, kernel_regularizer=l2)

        self.post_dense = Dense(64, kernel_regularizer=l2, activation='relu', name='post_dense')
        self.output_layer = Dense(num_classes, kernel_regularizer=l2, activation='softmax', name='output')
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
        self.batch_norm3 = BatchNormalization()

        self.dropout1 = Dropout(0.4)
        self.dropout2 = Dropout(0.3)
    
    def call(self, inputs, training=False):
        x1 = self.dense_input(inputs)
        x1 = self.batch_norm1(x1, training=training)
        x1 = self.dense1(x1)
        x1 = self.dropout1(x1)
        x2 = self.dense2(x1)
        x3 = self.dense3(x1)
        
        x = tf.concat([x2,x3], axis=1, name="concat")

        x = self.batch_norm2(x, training=training)
        x = tf.expand_dims(x, axis=1)
        x = self.gru(x)
        x = self.dropout2(x)
        x = self.post_dense(x)
        x = self.batch_norm3(x, training=training)
        return self.output_layer(x)


#Only to get the model summary
# model = SignLanguageModel(3)
# model.build((None, 63))
# model.summary()