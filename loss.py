
from tensorflow.keras import backend as K


def euclidean_distance(vects):
    (featsA, featsB) = vects
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		       keepdims=True)
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def euclidean_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(
        K.maximum(margin - y_pred, 0)))
