from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, concatenate
from keras.optimizers import RMSprop
from keras import backend as K
from keras import regularizers
import keras.optimizers as optimizers

def create_base_network(input_dim):
    a = 'tanh'
    model = Sequential()
    model.add(Dense(input_dim, input_shape=(input_dim, ), activation=a))
    model.add(Dense(600, activation=a))
    model.add(Dense(input_dim, activation=a))

    return model

def triplet_loss(y_true, y_pred):
    anchor = y_pred[:, 0:300]
    positive = y_pred[:, 300:600]
    negative = y_pred[:, 600:900]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    alpha = 1
    basic_loss = pos_dist - neg_dist + alpha

    loss = K.mean(basic_loss)

    return loss

def lossless_triplet_loss(y_true, y_pred):
    anchor = y_pred[:, 0:300]
    positive = y_pred[:, 300:600]
    negative = y_pred[:, 600:900]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    beta = 300.0
    epsilon = 1e-6

    pos_dist = pos_dist/beta
    neg_dist = neg_dist/beta
    return pos_dist - neg_dist + beta
    #pos_dist = -K.log(-(pos_dist/beta) + 1 + epsilon)
    #neg_dist = -K.log(-((beta-neg_dist)/beta) + 1 + epsilon)

    return pos_dist + neg_dist

def _pairwise_distances(y_pred, squared=False):
    dot_product = K.matmul(y_pred, K.transpose(y_pred))

    square_norm = K.diag_part(dot_product)

    distances = K.expand_dims(square_norm, 0) - 2.0 * dot_product + K.expand_dims(square_norm, 1)

    distances = K.maximum(distances, 0.0)

    if not squared:
        mask = K.to_float(K.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = K.sqrt(distances)

        distances = distances * (1.0 - mask)

    return distances

def _get_anchor_positive_triplet_mask(y_true):
    indices_equal = K.cast(K.eye(K.shape(labels)[0]), K.bool)
    indices_not_equal = K.logical_not(indices_equal)

    labels_equal = K.equal(K.expand_dims(labels, 0), K.expand_dims(labels, 1))

    mask = K.logical_and(indices_not_equal, labels_equal)

    return mask

def _get_anchor_negative_triplet_mask(y_true):
    labels_equal = K.equal(K.expand_dims(labels, 0), K.expand_dims(labels, 1))

    mask = K.logical_not(labels_equal)

    return mask

def _get_triplet_mask(y_true):
    indices_equal = K.cast(K.eye(K.shape(labels)[0]), K.bool)
    indices_not_equal = K.logical_not(indices_equal)
    i_not_equal_j = K.expand_dims(indices_not_equal, 2)
    i_not_equal_k = K.expand_dims(indices_not_equal, 1)
    j_not_equal_k = K.expand_dims(indices_not_equal, 0)

    distinct_indices = K.logical_and(K.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = K.equal(K.expand_dims(labels, 0), K.expand_dims(labels, 1))
    i_equal_j = K.expand_dims(label_equal, 2)
    i_equal_k = K.expand_dims(label_equal, 1)

    valid_labels = K.logical_and(i_equal_j, K.logical_not(i_equal_k))

    # Combine the two masks
    mask = K.logical_and(distinct_indices, valid_labels)

    return mask

def batch_all_triplet_loss(y_true, y_pred, margin, squared=False):
    pairwise_dist = _pairwise_distances(y_pred, squared=squared)

    anchor_positive_dist = K.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = K.expand_dims(pairwise_dist, 1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(labels)
    mask = K.to_float(mask)
    triplet_loss = K.multiply(mask, triplet_loss)

    triplet_loss = K.maximum(triplet_loss, 0.0)

    valid_triplets = K.to_float(K.greater(triplet_loss, 1e-16))
    num_positive_triplets = K.reduce_sum(valid_triplets)
    num_valid_triplets = K.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = K.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_postive_triplets

def batch_hard_triplet_loss(y_true, y_pred, margin=1, squared=False):
    pairwise_dist = _pairwise_distances(y_pred, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = K.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = K.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = K.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    K.summary.scalar("hardest_positive_dist", K.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = K.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = K.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = K.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    K.summary.scalar("hardest_negative_dist", K.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = K.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = K.reduce_mean(triplet_loss)

    return triplet_loss
