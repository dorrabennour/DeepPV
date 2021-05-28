import os
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import tensorflow.keras
from keras import backend as K
from keras import optimizers as opt
from keras import callbacks as cbks
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,UpSampling2D, Lambda,Cropping3D
from keras.layers import LeakyReLU
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Conv3D,Conv3DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D,MaxPooling3D,GlobalMaxPool3D
from keras.layers.merge import concatenate, add
from tensorflow.keras.models import model_from_yaml
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam, Adadelta,SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
path='/homedata/dbennour/DeepPV/Unet_models/'

def saveModel(model, savename,path):
#     path='/home/dbennour/DeepPV/Unet_models/' 
  # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(path+savename+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
        print( "Yaml Model ",savename,".yaml saved to disk")
        # serialize weights to HDF5
        model.save_weights(path+savename+".h5")
    print ("Weights ",savename,".h5 saved to disk")

def loadModel(savename,path):
#   path='/home/dbennour/DeepPV/Unet_models/'
  with open(path+savename+".yaml", "r") as yaml_file:
    model = model_from_yaml(yaml_file.read())
    print("Yaml Model ",savename,".yaml loaded ")
    model.load_weights(path+savename+".h5")
    print("Weights ",savename,".h5 loaded ")
  return model

def root_mean_squared_error(x_true, x_pred):
  from keras import backend as K
  return K.sqrt(K.mean(K.square(x_pred - x_true)))


def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv3D(filters = n_filters, kernel_size = (kernel_size, kernel_size,kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv3D(filters = n_filters, kernel_size = (kernel_size, kernel_size,kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


def R2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def Unet_build(n_filters = 16, dropout = 0, batchnorm = True):
    input_img = Input((513, 513, 2,1))
#     lr_schedule = ExponentialDecay(
#         initial_learning_rate=1e-4,
#         decay_steps=100,
#         decay_rate=0.9)
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 4, batchnorm = batchnorm)
    p1 =  MaxPooling3D((2, 2,2), padding = 'same')(c1)
    p1 = Dropout(dropout)(p1)


    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 4, batchnorm = batchnorm)
    p2 = MaxPooling3D((2, 2,2), padding = 'same')(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 4, batchnorm = batchnorm)
    p3 = MaxPooling3D((2, 2,2), padding = 'same')(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 4, batchnorm = batchnorm)
    p4 = MaxPooling3D((2, 2,1), padding = 'same')(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 4, batchnorm = batchnorm)

    # Expansive Path
    u6 = Conv3DTranspose(n_filters * 8, (3, 3,3), strides = (2, 2,1), padding = 'same')(c5)
    u6=Cropping3D(cropping=((1, 0), (1, 0),(0,0)), data_format=None)(u6)

    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 4, batchnorm = batchnorm)

    u7 = Conv3DTranspose(n_filters * 4, (3, 3,3), strides = (2, 2,1), padding = 'same')(c6)
    u7=Cropping3D(cropping=((1, 0), (1, 0),(0,0)), data_format=None)(u7)

    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 4, batchnorm = batchnorm)

    u8 = Conv3DTranspose(n_filters * 2, (3, 3,3), strides = (2, 2,1), padding = 'same')(c7)
    u8=Cropping3D(cropping=((1, 0), (1, 0),(0,0)), data_format=None)(u8)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 4, batchnorm = batchnorm)

    u9 = Conv3DTranspose(n_filters * 1, (3, 3,3), strides = (2, 2,2), padding = 'same')(c8)
    u9=Cropping3D(cropping=((1, 0), (1, 0),(0,0)), data_format=None)(u9)

    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    outputs = Conv3D(1, (1, 1,1), activation='tanh')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    adam=Adam(clipvalue=0.5)
    adadelta=Adadelta(clipvalue=0.5)
    sgd=SGD(momentum=0.9, nesterov=False)
    model.compile(optimizer=adam, loss=root_mean_squared_error, metrics=['mse',R2])
    
    
    return model



def Unet_fit(model,train_data,val_data,epoch):
    results_dir='/net/nfs/ssd1/dbennour/Resultats_ModelPV/'
    callbacks = [ EarlyStopping(patience=10, verbose=1),
                  ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
                  cbks.ModelCheckpoint(results_dir+'Unet-weights.h5', 
                                       monitor='val_loss', save_best_only=True),
                  TensorBoard(log_dir='/home/dbennour/tmp/Unet')]
    history=model.fit(train_data,epochs=epoch, callbacks=callbacks,validation_data=val_data)
    return history, model




def test_data(start,end,model_ae):

    test_list=sorted(os.listdir('/net/nfs/ssd1/dbennour/CAL_CMV_slot/test_data/'))[start:end]
    dirout_test='/net/nfs/ssd1/dbennour/CAL_CMV_slot/test_data/'
    X_test=np.empty((len(test_list),513,513,2,1))
    X_test_ae=np.empty((len(test_list),513,513,2,1))
    Y_test=np.empty((len(test_list),513,513,2,1))
    for i,file in enumerate(test_list):
        file_test=xr.open_dataset(dirout_test+file)
        X_test_T0=file_test[['CAL_T0']].to_array().values
        X_test_T015=file_test[['CAL_T0-15']].to_array().values
#         print(X_test_T0.shape)
        X_test[i,:,:,0,:]=X_test_T0.reshape((-1,513,513,1))
        X_test[i,:,:,1,:]=X_test_T015.reshape((-1,513,513,1))
        
        X_test_T0_ae=model_ae.predict(X_test_T0)
        X_test_T015_ae=model_ae.predict(X_test_T015)
        
        Y_test_x=file_test[['CMV_X']].to_array().values
        Y_test_y=file_test[['CMV_Y']].to_array().values
        
        X_test_ae[i,:,:,0,:]=X_test_T0_ae.reshape((-1,513,513,1))
        X_test_ae[i,:,:,1,:]=X_test_T015_ae.reshape((-1,513,513,1))
        Y_test[i,:,:,0,:]=Y_test_x.reshape((-1,513,513,1))
        Y_test[i,:,:,1,:]=Y_test_y.reshape((-1,513,513,1))
        return X_test_ae,Y_test,X_test