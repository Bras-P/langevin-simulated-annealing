import tensorflow as tf

class NBatchValLogger(tf.keras.callbacks.Callback):
    def __init__(self, N, model, x_test, y_test):
        super().__init__()
        self.history = {'val_accuracy':[],}
        self.N = N
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
#
    def on_batch_end(self, batch, logs={}):
        if batch%self.N==0:
            self.history['val_accuracy'].append(self.model.evaluate(self.x_test, self.y_test, batch_size=32, verbose=0)[1])
