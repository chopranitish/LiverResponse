__author__ = 'Brian M Anderson'
# Created on 11/18/2020

from DL_Template.ReturnGenerators import return_paths, return_generators, plot_scroll_Image
from DL_Template.ReturnCosineLoss import CosineLoss, cosine_loss
from DL_Template.ReturnModels import return_model
import tensorflow as tf
import os
import numpy as np

loss = CosineLoss()
model_key = 3
id = 1
base_path, morfeus_drive, train_generator, validation_generator = return_generators(evaluate=False, batch_size=16,
                                                                                    cache=False, cross_validation_id=id,
                                                                                    model_key=model_key)
x, y = next(iter(validation_generator.data_set))
model_path_dir = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records\Models\Model_2{}'.format(id)
model_path = os.path.join(model_path_dir, 'cp.ckpt')
tf_path = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records\Tensorboard\TB_2{}'.format(id)
model = return_model(model_key=model_key)
if model_key > 2:
    model = model()
# if not os.path.exists(model_path_dir) or not os.listdir(model_path_dir):
    # loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(lr=1e-3)
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True,
                                                save_freq='epoch', save_weights_only=True, verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tf_path,# profile_batch='50,100',
                                             write_graph=True)  # profile_batch='300,401',
callbacks = [tensorboard]
model.compile(optimizer, loss=loss, metrics=['accuracy'])
# model.train_on_batch(x, y)
# pred = model.predict(x)
# cosine_loss(y_true=y, y_pred=pred)
model.fit(train_generator.data_set, epochs=30, steps_per_epoch=len(train_generator),
          validation_data=validation_generator.data_set, validation_steps=len(validation_generator),
          validation_freq=1, callbacks=callbacks)

# model.load_weights(model_path)
# xxx = 1
# pred_list = []
# truth_list = []
# val_iterator = iter(validation_generator.data_set)
# for i in range(len(validation_generator)):
#     print(i)
#     x, y = next(val_iterator)
#     pred = model.predict(x)
#     output_pred = np.argmax(pred)
#     pred_list.append(output_pred)
#     truth_list.append(np.argmax(y[0].numpy()))
# print(truth_list)
# print(pred_list)