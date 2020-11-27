
####################
#resnet_v2
####################

#%%
import keras
from keras import backend as K 
from keras import losses,optimizers
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential,load_model,Model
from keras.layers import Layer,Input,AveragePooling2D,Reshape,Lambda,Add,Permute
from keras.layers import GlobalAveragePooling2D,concatenate,Multiply,Embedding
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D,MaxPooling2D,Conv2DTranspose
from keras.regularizers import l2
from keras.utils.data_utils import get_file
from keras.utils import to_categorical
from keras.engine.topology import get_source_inputs
import numpy as np

def se_block(x,num_filters=16,ratio=16):
    """これはなんのブロック？"""
    se = GlobalAveragePooling2D()(x)
    se = Dense(num_filters//ratio,activation="relu",kernel_initializer="he_normal",
                use_bias=False)(se)
    se = Dense(num_filters,activation="sigmoid",use_bias=False)(se)
    return Multiply()([se,x])

def resnet_layer(inputs,
                num_filters=16,
                kernel_size=3,
                strides=1,
                activation="relu",
                batch_normalization=True,
                conv_first=True,
                use_se_block=False,
                add_last_bn=False
                ):
    """
    2D convolutional-batch normalization activation tack builder
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding="same",
                  kernel_initializer="he_normal",
                  kernel_regularizer=l2(1e-4))
    
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    if add_last_bn:
        x = BatchNormalization()(x)
    if use_se_block:
        x = se_block(x,num_filters=num_filters)
    
    return x

def resnet_v2(input_shape,depth,num_classes=10):
    """resnet_v2"""

    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2")

    num_filters_in = 16
    num_res_blocks = int(((depth - 2) / 9))
    

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)
    
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = "relu"
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2
            
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            
            if res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_in,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=batch_normalization)
            x = keras.layers.add([x, y])
        
        num_filters_in = num_filters_out
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation="softmax",
                    kernel_initializer="he_normal")(y)
    model = Model(inputs=inputs,outputs=outputs)
    return model

def resnet_with_center_loss(input_shape,depth,num_classes=10):
    """resnetにcenter lossを追加してみる。
    input_shape：画像の入力形式
    """
    if (depth - 2) % 9 != 0:
        raise ValueError("depth should be 9n+2")

    num_filters_in = 16
    num_res_blocks = int(((depth - 2) / 9))
    

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)
    
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = "relu"
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2
            
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            
            if res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
        
        num_filters_in = num_filters_out
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #x = AveragePooling2D(pool_size=8)(x)
    x = AveragePooling2D(pool_size=4)(x)
    #ここからcenter-lossを使う場合
    #y = Flatten(name="feature_out")(x)
    x = Flatten()(x)
    y = Dense(2,name="feature_out")(x)
    main = Dense(num_classes,
                 activation="softmax",
                 kernel_initializer="he_normal",
                 name="main_out")(y)
    
    label_inputs = Input((num_classes,))
    side = CenterLossLayer(alpha=0.5, name='centerlosslayer')([y, label_inputs])
    
    model = Model(inputs=[inputs, label_inputs], outputs=[main, side])
    return model

class CenterLossLayer(Layer):
    """center_lossの層
    参考
    https://github.com/handongfeng/MNIST-center-loss/blob/master/centerLoss_MNIST.py
    """
    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(10, 2),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):

        # x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True) #/ K.dot(x[1], center_counts)
        return self.result # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)


if __name__ == "__main__":
    from keras.datasets import mnist
    from sklearn.metrics import accuracy_score
    # mnistで試す。
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_train_onehot = to_categorical(y_train, 10)#one hot encoding
    y_test_onehot = to_categorical(y_test, 10)#one hot encoding

    lambda_c = 0.2 #ここのパラメータは調整必要っぽい。
    input_shape = (28,28,1)
    model = resnet_with_center_loss(input_shape=input_shape,
                                    depth=20)#ここはあとで確認する。
    optim = optimizers.SGD(lr=1e-3, momentum=0.9)#仮
    model.compile(optimizer=optim,#ここに加えること。
                  loss=[losses.categorical_crossentropy, zero_loss],
                  loss_weights=[1, lambda_c])

    #初期のcenter
    dummy1 = np.zeros((x_train.shape[0], 1))
    dummy2 = np.zeros((x_test.shape[0], 1))

    model.fit([x_train, y_train_onehot], [y_train_onehot, dummy1], 
              batch_size=16,
              epochs=10,
              verbose=2, 
              validation_data=([x_test, y_test_onehot], [y_test_onehot, dummy2]),
              )

    #推論処理する場合
    model_for_predict = Model(inputs=model.inputs[0],outputs=model.outputs[0])
    predicted_vec = model_for_predict.predict(x_test)
    predicted_label = np.argmax(predicted_vec,axis=0)

    #accuracyを出す。
    print(accuracy_score(y_test,predicted_label))
    
# %%
