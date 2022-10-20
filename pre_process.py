import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class data_prep:
    def __init__(self,series,window_size, buffer=100, batch_size=64,split_ratio=0.8):
        self.series = series
        self.window_size = window_size
        self.buffer = buffer
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.train_test_split(self.series)

    
    def train_test_split(self,series):
        self.train = series[:int(self.split_ratio*len(series))]
        self.test = series[int(self.split_ratio*len(series)):]

    def pre_process(self,train):
        if train:
            series = tf.expand_dims(self.train, axis=-1)
            self.scaler = MinMaxScaler(feature_range=(0,1))
            series = self.scaler.fit_transform(series)
            ds = tf.data.Dataset.from_tensor_slices(series)
            ds = ds.window(size=self.window_size + 1, shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda win: win.batch(self.window_size + 1))
            ds = ds.shuffle(self.buffer)
            ds = ds.map(lambda win: (win[: -1], win[-1]))
        
        else:
            series = tf.expand_dims(self.test, axis=-1)
            series = self.scaler.transform(series)
            ds = tf.data.Dataset.from_tensor_slices(series)
            ds = ds.window(size=self.window_size , shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda win: win.batch(self.window_size))

        ds = ds.batch(self.batch_size).prefetch(1)

        return ds

    



