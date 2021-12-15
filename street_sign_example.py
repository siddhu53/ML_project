import os
import glob
from posixpath import basename
from sklearn.model_selection  import train_test_split
import shutil
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

from my_utils import create_generators, split_data, order_test_set

from deeplearning_models import streetsigns_model
import tensorflow as tf


if __name__ == "__main__":
    
    if False:
        
        path_to_data = "F:\\tensorflow-tutorial\\traffic_light_dataset\\German_trafficlight_dataset\\Train"
        path_to_save_training = "F:\\tensorflow-tutorial\\traffic_light_dataset\\German_trafficlight_dataset\\training_data\\train"
        path_to_save_validation = "F:\\tensorflow-tutorial\\traffic_light_dataset\\German_trafficlight_dataset\\training_data\\val"
        split_data(path_to_data,path_to_save_training,path_to_save_validation)        
    if False:
        path_to_images = "F:\\tensorflow-tutorial\\traffic_light_dataset\\German_trafficlight_dataset\\Test"
        path_to_csv = "F:\\tensorflow-tutorial\\traffic_light_dataset\\German_trafficlight_dataset\\Test.csv"
        order_test_set(path_to_images,path_to_csv)
        
path_to_training = "F:\\tensorflow-tutorial\\traffic_light_dataset\\German_trafficlight_dataset\\training_data\\train"
path_to_validation = "F:\\tensorflow-tutorial\\traffic_light_dataset\\German_trafficlight_dataset\\training_data\\val"
path_to_test = "F:\\tensorflow-tutorial\\traffic_light_dataset\\German_trafficlight_dataset\\Test"
batch_size = 64
epochs=16
lr = 0.0001

train_generator,val_generator,test_generator = create_generators(batch_size,path_to_training,path_to_validation,path_to_test)   
nbr_classes = train_generator.num_classes

TRAIN=False
TEST =True



path_to_save_model = './Models'
ckpt_saver = ModelCheckpoint(
path_to_save_model,
monitor = "val_accuracy",
mode='max',
save_best_only =True,
save_freq = 'epoch',
verbose=1
    )

early_stop = EarlyStopping(monitor="val_accuracy",patience=10)


model = streetsigns_model(nbr_classes)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr,amsgrad=True)
    
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_generator,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=val_generator,
          callbacks = [ckpt_saver,early_stop]
          )

if TEST:
    model = tf.keras.models.load_model('./Models')
    model.summary()
    
    print('Evaluating validation set..')
    model.evaluate(val_generator)

    print('Evaluating test set..')
    model.evaluate(test_generator)
