
import os
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, MaxPooling2D, \
    Dense, Dropout, Activation, Input, Lambda, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop

from data import CFRDatagenerator, plot_pairs
from loss import euclidean_distance, euclidean_dist_output_shape, \
    contrastive_loss

from mobilenet import build_mobilefacenet


dataset_dir = '/home/ankit/disk/skylark/cfp-dataset/Data/'
training_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')
model_path = '/home/ankit/disk/skylark/models/mobilenet_cfp/model.h5'

epochs = 20
steps_per_epoch = 1000
# input_shape = (100, 100, 3)
input_shape = (112, 96, 3)
batch_size = 8
val_steps = 100
assert batch_size % 2 == 0, 'Batch size has to be even. batch_size={}'.format(
    batch_size)


traingen = CFRDatagenerator(training_dir, input_dims=input_shape,
                            batch_size=batch_size, no_of_batches_per_epoch=steps_per_epoch)
valgen = CFRDatagenerator(validation_dir, input_dims=input_shape,
                          batch_size=batch_size, no_of_batches_per_epoch=val_steps)

# example plots
# for pairs, labels in valgen:
#     plot_pairs(pairs, labels)
#     import pdb; pdb.set_trace()

base_model = build_mobilefacenet(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

embeddings_a = base_model(input_a)
embeddings_b = base_model(input_b)

distance = Lambda(euclidean_distance, output_shape=euclidean_dist_output_shape
                  )([embeddings_a, embeddings_b])

model = Model(inputs=[input_a, input_b], outputs=distance)
opt = RMSprop()
model.compile(loss=contrastive_loss, optimizer=opt)

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()


model.fit(traingen, validation_data=valgen, validation_steps=val_steps,
          steps_per_epoch=steps_per_epoch, epochs=epochs)

# preds = model.predict(valgen)
model.save(model_path)

# compute accuracy
val_step_count = 0
predictions = []
labels = []
for X, y in valgen:
    preds = model.predict(X)
    preds = np.squeeze(preds)
    predictions = predictions + preds.tolist()
    labels = labels + y.tolist()
    val_step_count += 1
    if val_step_count > val_steps:
        break

predictions = np.array(predictions)
labels = np.array(labels)
accuracy = compute_accuracy(predictions, labels)
print('accuracy: {}'.format(accuracy))

# for pairs, labels in traingen:
#     plot_pairs(pairs, labels)







