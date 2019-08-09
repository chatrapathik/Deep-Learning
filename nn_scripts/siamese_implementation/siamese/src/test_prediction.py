import json

import numpy as np
import csv

from basescript import BaseScript
from diskarray import DiskArray

from keras import models, layers, optimizers, metrics
from keras.models import load_model
import keras.backend as K

def _euclidean_distance(vects):
    x, y = vects
    #return K.sqrt(K.sum(K.square(x - y), keepdims=True))
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def _eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

class TestPrediction(BaseScript):
    def run(self):
        model = load_model(self.args.model_name, custom_objects=dict(
                    _euclidean_distance=_euclidean_distance,
                    _eucl_dist_output_shape=_eucl_dist_output_shape
                )
            )

        test_d = DiskArray(self.args.test_f, dtype=[('vec', np.float32, 300), ('vec', np.float32, 300), ('label', np.int)])
        csv_f = open(self.args.csv_file, 'w')
        csv_file = csv.writer(csv_f)
        csv_file.writerow(['label', 'prediction'])
        for i in len(test_d['vec']):
            vec1 = test_d['vec1'][i]
            vec2 = test_d['vec2'][i]
            pred_val = self.get_prediction(vec1, vec2, model)
            label = test_d['label'][i]
            csv_file.writerow([label, pred_val])

    def get_prediction(self, vec1, vec2, model):
        vec1 = vec1.reshape(1, len(vec1))
        vec2 = vec2.reshape(1, len(vec2))
        p_value = model.predict(vec1, vec2)
        return p_value[0][0]

    def define_args(self, parser):
        parser.add_argument('test_f', help='input json file')
        parser.add_argument('model_name', help='model name')
        parser.add_argument('csv_file', help='csv file to store pred val')

if __name__ == '__main__':
    TestPrediction().start()
