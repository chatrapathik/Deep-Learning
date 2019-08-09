from sys import argv as A

import json
import requests

import numpy as np
import pandas as pd

import tornado.ioloop
import tornado.web

from scipy.spatial import distance
from wordvecspace import WordVecSpaceMem
from diskarray import DiskArray


from typing import Union
from kwikapi import API
from kwikapi.tornado import RequestHandler

class KNearestService(object):
    def __init__(self, actual_vspace, transformed_vspace):

        self.wvspace = WordVecSpaceMem(actual_vspace)
        self.t_vspace = DiskArray(transformed_vspace, dtype=[('vec', np.float32, 300)])


    def k_nearest(self, word: str, k: int=10, metric: str='angular') -> dict:
        index = self.wvspace.get_word_index(word)

        result = self.wvspace.get_nearest(index, k, metric=metric)
        actual_results = self.wvspace.get_word_at_indices(result)

        vec = self.t_vspace['vec'][index].reshape(1, 300)
        vecs = self.t_vspace['vec']

        if metric == 'angular':
            metric = 'cosine'

        dist = distance.cdist(vec, vecs, metric)

        dist = pd.Series(dist[0])
        res = dist.nsmallest(k).keys()
        trans_results = self.wvspace.get_word_at_indices(list(res))

        recall = len(set(actual_results) & set(trans_results)) / k

        data = dict(vspace_results=actual_results, T_vspace_results=trans_results, recall=recall)
        return data


def make_app():
    return tornado.web.Application(
        [
            (r'^/api/.*', RequestHandler, dict(api=api)),
        ]
    )

if __name__ == "__main__":
    api = API()
    api.register(KNearestService(A[1], A[2]), 'v1')

    app = make_app()
    app.listen(A[3])

    tornado.ioloop.IOLoop.current().start()
