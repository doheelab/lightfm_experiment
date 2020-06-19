#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:49:40 2020

@author: dohee
"""

import os
os.chdir(r"/home/dohee/kaggle/lightfm-paper")

# LightFM(tags), Cold, 0.661
#!ipython -- experiments/movielens/model.py --dim 50 --cold --tags --split 0.2 

# LightFM(tags + ids), Cold, 0.683 
#!ipython -- experiments/movielens/model.py --dim 50 --cold --tags --ids --split 0.2 

# LSI-LR, Cold, 0.690-695
#!ipython -- experiments/movielens/model.py --dim 50 --lsi --cold --tags --ids --split 0.2 


#%%

import argparse
import json
import logging
import numpy as np
from pprint import pformat
import scipy.sparse as sp
import sys

from lightfm import LightFM

import experiments
from experiments.cf_model import CFModel
from experiments.lsiup_model import LsiUpModel
from experiments.modelutils import fit_model
from experiments.movielens.data import read_movie_features, read_interaction_data

logger = experiments.getLogger('experiments.movielens.model')

#%%

def read_data(titles, genres,
              genome_tag_threshold,
              positive_threshold):

    print('Reading features')
    features = read_movie_features(titles=titles, genres=genres, genome_tag_threshold=genome_tag_threshold)
    item_features_matrix = features.mat.tocoo().tocsr()

    print('Reading interactions')
    interactions = read_interaction_data(features.item_ids,
                                         positive_threshold=positive_threshold)
    interactions.fit(min_positives=1, sampled_negatives_ratio=0, use_observed_negatives=True)

    print('%s users, %s items, %s interactions, %s item features in the dataset',
                len(interactions.user_ids), len(features.item_ids),
                len(interactions.data), len(features.feature_ids))

    return features, item_features_matrix, interactions


def run(features,
        item_features_matrix,
        interactions,
        cf_model,
        lsiup_model,
        n_iter,
        test_size,
        cold_start,
        learning_rate,
        no_components,
        a_alpha,
        b_alpha,
        epochs):

    logger.debug('Fitting the model with %s', locals())

    no_interactions = len(interactions.data)

    if cf_model:
        logger.info('Fitting the CF model')
        modelfnc = lambda: CFModel(dim=no_components)
    elif lsiup_model:
        logger.info('Fitting the LSI-UP model')
        modelfnc = lambda: LsiUpModel(dim=no_components)
    else:
        modelfnc = lambda: LightFM(learning_rate=learning_rate,
                                    no_components=no_components,
                                    item_alpha=a_alpha,
                                    user_alpha=b_alpha)

    model, auc = fit_model(interactions=interactions,
                           item_features_matrix=item_features_matrix, 
                           n_iter=n_iter,
                           epochs=epochs,
                           modelfnc=modelfnc,
                           test_size=test_size,
                           cold_start=cold_start)
    logger.debug('Average AUC: %s', auc)

    if not cf_model and not lsiup_model:
        model.add_item_feature_dictionary(features.feature_ids, check=False)
        features.add_latent_representations(model.item_features)

        titles = ('Lord of the Rings: The Two Towers, The (2002)',
                  'Toy Story (1995)',
                  'Terminator, The (1984)',
                  'Europa Europa (Hitlerjunge Salomon) (1990)')

        for title in titles:
            logger.debug('Most similar movies to %s: %s', title,
                        features.most_similar_movie(title, number=20))

            # Can only get similar tags if we have tag features
        test_features = ('genome:art house',
                         'genome:dystopia',
                         'genome:bond')

        for test_feature in test_features:
            try:
                logger.debug('Features most similar to %s: %s',
                             test_feature,
                             model.most_similar(test_feature, 'item', number=10))
            except KeyError:
                pass

    return auc

#%%

import array, collections
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
class StratifiedSplit(object):
    """
    Class responsible for producing train-test splits.
    """

    def __init__(self, user_ids, item_ids, n_iter=10, 
                 test_size=0.2, cold_start=False, random_seed=None):
        """
        Options:
        - test_size: the fraction of the dataset to be used as the test set.
        - cold_start: if True, test_size of items will be randomly selected to
                      be in the test set and removed from the training set. When
                      False, test_size of all training pairs are moved to the
                      test set.
        """

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.no_interactions = len(self.user_ids)
        self.n_iter = n_iter
        self.test_size = test_size
        self.cold_start = cold_start

        self.shuffle_split = ShuffleSplit(test_size=self.test_size)
        #self.shuffle_split = ShuffleSplit(self.no_interactions,
        #                                  n_iter=self.n_iter,
        #                                  test_size=self.test_size)

    def _cold_start_iterations(self):
        """
        Performs the cold-start splits.
        """

        for _ in range(self.n_iter):
            unique_item_ids = np.unique(self.item_ids)
            no_in_test = int(self.test_size * len(unique_item_ids))

            item_ids_in_test = set(np.random.choice(unique_item_ids, size=no_in_test))

            test_indices = array.array('i')
            train_indices = array.array('i')

            for i, item_id in enumerate(self.item_ids):
                if item_id in item_ids_in_test:
                    test_indices.append(i)
                else:
                    train_indices.append(i)

            train = np.frombuffer(train_indices, dtype=np.int32)
            test = np.frombuffer(test_indices, dtype=np.int32)

            # Shuffle data.
            np.random.shuffle(train)
            np.random.shuffle(test)

            yield train, test

    def __iter__(self):

        #if self.cold_start:
        #    splits = self._cold_start_iterations()           
        #else:
        #    splits = self.shuffle_split
        
        splits = self._cold_start_iterations()

        for train, test in splits:

            # Make sure that all the users in test
            # are represented in train.
            user_ids_in_train = collections.defaultdict(lambda: 0)
            item_ids_in_train = collections.defaultdict(lambda: 0)

            for uid in self.user_ids[train]:
                user_ids_in_train[uid] += 1

            for iid in self.item_ids[train]:
                item_ids_in_train[iid] += 1

            if self.cold_start:
                test = [x for x in test if self.user_ids[x] in user_ids_in_train]
            else:
                # For the non-cold start scenario, make sure that both users
                # and items are represented in the train set.
                test = [x for x in test if (self.user_ids[x] in user_ids_in_train
                                            and self.item_ids[x] in item_ids_in_train)]

            test = np.array(test)

            yield train, test
        
def build_user_feature_matrix(user_ids):

    n = len(user_ids)

    return sp.coo_matrix((np.ones(n, dtype=np.int32), (np.arange(n), user_ids))).tocsr()


def stratified_roc_auc_score(y, yhat, user_indices):
    """
    Compute ROC AUC for each user individually, then average.
    """

    aucs = []

    y_dict = collections.defaultdict(lambda: array.array('d'))
    yhat_dict = collections.defaultdict(lambda: array.array('d'))

    for i, uid in enumerate(user_indices):
        y_dict[uid].append(y[i])
        yhat_dict[uid].append(yhat[i])

    for uid in y_dict:

        user_y = np.frombuffer(y_dict[uid], dtype=np.float64)
        user_yhat = np.frombuffer(yhat_dict[uid], dtype=np.float64)

        if len(user_y) and len(user_yhat) and len(np.unique(user_y)) == 2:
            aucs.append(roc_auc_score(user_y, user_yhat))

    print('%s users in stratified ROC AUC evaluation.', len(aucs))
    
    return np.mean(aucs)

#%%


#%%
"""
parser = argparse.ArgumentParser(description='Run the MovieLens experiment')
parser.add_argument('-i', '--ids', action='store_true',
                  help='Use item ids as features.')
parser.add_argument('-t', '--tags', action='store_true',
                    help='Use tags as features.')
parser.add_argument('-s', '--split', action='store', required=True, type=float,
                    help='Fraction (eg, 0.2) of data to use as the test set')
parser.add_argument('-c', '--cold', action='store_true',
                    help='Use the cold start split.')
parser.add_argument('-l', '--lsi', action='store_true',
                    help='Use the LSI-LR model')
parser.add_argument('-u', '--up', action='store_true',
                    help='Use the LSI-UP model')
parser.add_argument('-d', '--dim', action='store',
                    type=int, default=(64,),
                    nargs='+',
                    help='Latent dimensionality of the model.')
parser.add_argument('-n', '--niter', action='store',
                    type=int, default=5,
                    help='Number of train/test splits')
"""
#args = parser.parse_args()

#logger.info('Running the MovieLens experiment.')
#logger.info('Configuration: %s', pformat(args))

cold=True
dim=[50]
ids=False # ids=False
tags=True
lsi=False
niter=5
split=0.2
up=False
# A large tag threshold excludes all tags.
tag_threshold = 0.8 #if args.tags else 100.0

# features, item_features_matrix : movie - genre info (10681*1010)
# interactions : self.data, self.user_id, self.item_id (9996948,)

features, item_features_matrix, interactions = read_data(titles=ids,
                                                         genres=False,
                                                         genome_tag_threshold=tag_threshold,
                                                         positive_threshold=4.0)

#titles=ids
#genres=False
#genome_tag_threshold=tag_threshold
#positive_threshold=4.0
#a = read_movie_features(titles=titles, genres=genres, genome_tag_threshold=genome_tag_threshold)

results = {}
cf_model=False
lsiup_model=False
n_iter=5
test_size=0.2
cold_start=True
learning_rate=0.05
no_components=int(50)
a_alpha=0.0
b_alpha=0.0
epochs=30
no_interactions = len(interactions.data)
 
modelfnc = lambda: LightFM(learning_rate=learning_rate,
                            no_components=no_components,
                            item_alpha=a_alpha,
                            user_alpha=b_alpha)

"""
model, auc = fit_model(interactions=interactions,
                       item_features_matrix=item_features_matrix, 
                       n_iter=n_iter,
                       epochs=epochs,
                       modelfnc=modelfnc,
                       test_size=test_size,
                       cold_start=cold_start)
"""


#%%


# fit_model
kf = StratifiedSplit(interactions.user_id, interactions.item_id,
                     n_iter=n_iter, test_size=test_size, cold_start=cold_start)

# Store ROC AUC scores for all iterations.
aucs = []

# Iterate over train-test splits.
for i, (train, test) in enumerate(kf):

    print('Split no %s', i)
    print('%s examples in training set, %s in test set. Interaction density: %s',
                len(train), len(test), float(len(train)) / (len(interactions.user_ids)
                                                            * len(interactions.item_ids)))

    # For every split, get a new model instance.
    model = modelfnc()


    # LightFM and MF models using the LightFM implementation.
    
    user_features_matrix = None
    if user_features_matrix is not None:
        user_features = user_features_matrix
    else:
        user_features = build_user_feature_matrix(interactions.user_id)

    item_features = item_features_matrix

    previous_auc = 0.0

    interactions.data[interactions.data == 0] = -1

    train_interactions = sp.coo_matrix((interactions.data[train],
                                        (interactions.user_id[train],
                                         interactions.item_id[train])))

    # Run for a maximum of epochs epochs.
    # Stop if the test score starts falling, take the best result.
    for x in range(epochs):
        print("Epoch: %s", x)
        model.fit_partial(train_interactions,
                          item_features=item_features,
                          user_features=user_features,
                          epochs=1, num_threads=4)

        train_predictions = model.predict(interactions.user_id[train],
                                          interactions.item_id[train],
                                          user_features=user_features,
                                          item_features=item_features,
                                          num_threads=4)
        test_predictions = model.predict(interactions.user_id[test],
                                         interactions.item_id[test],
                                         user_features=user_features,
                                         item_features=item_features,
                                         num_threads=4)

        train_auc = stratified_roc_auc_score(interactions.data[train],
                                             train_predictions,
                                             interactions.user_id[train])
        test_auc = stratified_roc_auc_score(interactions.data[test],
                                            test_predictions,
                                            interactions.user_id[test])

        print('Epoch %s, test AUC %s, train AUC %s', x, test_auc, train_auc)

        if previous_auc > test_auc:
            print("pre: %s, current: %s", previous_auc, test_auc)
            break

        previous_auc = test_auc

    aucs.append(previous_auc)

## result
## 0.663066 (tags)
## 0.699208 (tags + ids)



#%%

train_predictions = model.predict(interactions.user_id[train],
                                  interactions.item_id[train],
                                  user_features=user_features,
                                  item_features=item_features,
                                  num_threads=4)

model.get_item_representations()[0].shape
model.get_item_representations()[1].shape
model.get_user_representations()[0].shape
model.get_user_representations()[1].shape
