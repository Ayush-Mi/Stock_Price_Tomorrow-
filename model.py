import tensorflow as tf


class forecast_model:
    def __init__(self,):
        self.build_model()
    
    def build_model(self,):
        model = tf.keras.models.Sequential([tf.keras.layers.Conv1D(filters=64, kernel_size=5,
                                                                padding="causal",
                                                                activation="relu",
                                                                input_shape=[None, 1]),
                                            tf.keras.layers.LSTM(64, return_sequences=True),
                                            tf.keras.layers.LSTM(128,return_sequences=True),
                                            tf.keras.layers.Dense(30, activation="relu"),
                                            tf.keras.layers.Dense(10, activation="relu"),
                                            tf.keras.layers.Dense(1),
        ])

        return model