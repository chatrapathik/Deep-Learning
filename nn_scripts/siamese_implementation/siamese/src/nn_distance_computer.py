'''Neural Network Distance Computer'''

import numpy as np
from scipy.spatial import distance
from keras import models, layers, optimizers
from keras import backend as K
from keras.models import load_model
from deepindex import BaseDistanceComputer

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(
            (y_true * K.square(y_pred)) +
            (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))
    )

def _euclidean_distance(vects):
    x, y = vects
    #return K.sqrt(K.sum(K.square(x - y), keepdims=True))
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def _eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

class NNDistanceComputer(BaseDistanceComputer):

    EPOCHS = 15
    HIDDEN_LAYERS = 2
    ACTIVATION = 'softplus'
    HIDDEN_NEURONS = 64
    OUTPUT_ACTIVATION = 'sigmoid'
    OPTIMIZER = 'rmsprop'
    LOSS = 'binary_crossentropy'
    METRICS = ['acc']

    def __init__(self, model_path=None, dimension=None):
        if model_path is not None:
            self.model = load_model(model_path,
                custom_objects=dict(
                    contrastive_loss=contrastive_loss,
                    _euclidean_distance=_euclidean_distance,
                    _eucl_dist_output_shape=_eucl_dist_output_shape)
            )
        else:
            self.dimension = int(dimension)
            self.model = models.Sequential()
            self.model.add(
                layers.Dense(
                    self.dimension + 1,
                    activation=self.ACTIVATION,
                    input_shape=(
                        self.dimension + 1,
                    )))
            for layer in range(self.HIDDEN_LAYERS):
                self.model.add(
                    layers.Dense(
                        self.HIDDEN_NEURONS,
                        activation=self.ACTIVATION))
                self.HIDDEN_NEURONS //= 2

            self.model.add(layers.Dense(1, activation=self.OUTPUT_ACTIVATION))
            self.model.compile(
                optimizer=self.OPTIMIZER,
                loss=self.LOSS,
                metrics=self.METRICS)

    def get_distance(self, v1, v2):
        # Calculating distance b/w two vectors
        '''
        diff_vector = np.zeros(len(v1))
        diff_vector[0:len(v1)] = np.abs(v1 - v2)
        diff_vector = np.insert(diff_vector, len(diff_vector), distance.cosine(v1, v2))
        '''
        '''
        Whenever we operate on 1D numpy array then the result is implicitly a tuple.
        But Neural Networks only operate upon numpy arrays. So, we need to
        reshape our output.
        '''

        #diff_vector = diff_vector.reshape(1, len(diff_vector))

        '''
        The value stored in the variable similarities will be a 2D numpy array.
        Since, we are calculating distance b/w two vectors, this variable will have only one value
        in the shape of [[_value_]]. That value can easily be accessed by similarities[0][0].
        '''
        v1 = v1.reshape(1, len(v1))
        v2 = v2.reshape(1, len(v2))

        similarities = self.model.predict([v1, v2])
        return similarities[0][0]

    def get_distances(self, vec, vecs):
        # Calculating distance b/w vector and vectors matrix
        '''
        diff_vectors = np.abs(vec - vecs)
        cosine_dist = distance.cdist(vec.reshape(1, len(vec)), vecs, 'cosine')
        '''
        '''
        The above operation gives 2D numpy array as an output. But we need the transpose of the output.
        So, we need to reshape our output.
        '''
        #cosine_dist = cosine_dist.reshape(len(vecs), 1)
        #tensors = np.append(diff_vectors, cosine_dist, axis=1)
        vec = vec.reshape(1, len(vec))
        similarities = self.model.predict([vec.repeat(len(vecs), axis=0), vecs])

        return similarities

    def get_distance_matrix(self, vectors):
        # Calculating within a vectors matrix
        distances = []
        for vector in vectors:
            distances.append(self.get_distances(vector, vectors))

        distances = np.asarray(distances)
        distances = np.reshape(distances, (len(distances), len(distances[0])))
        return distances
