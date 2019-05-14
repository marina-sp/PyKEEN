# -*- coding: utf-8 -*-

"""Implementation of the basic pipeline."""

import logging
from collections import OrderedDict
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import rdflib
import torch
from sklearn.model_selection import train_test_split
from torch.nn import Module

import pykeen.constants as pkc
from pykeen.hpo import RandomSearch
from pykeen.kge_models import get_kge_model
from pykeen.utilities.evaluation_utils.metrics_computations import MetricResults, compute_metric_results
from pykeen.utilities.train_utils import train_kge_model
from pykeen.utilities.triples_creation_utils import create_mapped_triples, create_mappings

__all__ = [
    'Pipeline',
    'load_data',
]

log = logging.getLogger(__name__)


class Pipeline(object):
    """Encapsulates the KGE model training pipeline."""

    def __init__(self, config: Dict):
        self.config: Dict = config
        self.seed: int = config[pkc.SEED] if pkc.SEED in config else 2
        self.entity_label_to_id: Dict[int: str] = None
        self.relation_label_to_id: Dict[int: str] = None
        self.device_name = (
            'cuda:0'
            if torch.cuda.is_available() and self.config[pkc.PREFERRED_DEVICE] == pkc.GPU else
            pkc.CPU
        )
        self.device = torch.device(self.device_name)

        # set num of entities
        _ = self._get_train_and_test_triples()

    @staticmethod
    def _use_hpo(config):
        return config[pkc.EXECUTION_MODE] == pkc.HPO_MODE

    @property
    def is_evaluation_required(self) -> bool:
        return pkc.TEST_SET_PATH in self.config or pkc.TEST_SET_RATIO in self.config

    def run(self) -> Mapping:
        """Run this pipeline."""
        metric_results = None

        if self._use_hpo(self.config):  # Hyper-parameter optimization mode
            mapped_pos_train_triples, mapped_pos_test_triples, mapped_neg_test_triples = \
                self._get_train_and_test_triples()

            (trained_model,
             loss_per_epoch,
             valloss_per_epoch,
             entity_label_to_embedding,
             relation_label_to_embedding,
             metric_results,
             params) = RandomSearch.run(
                mapped_train_triples=mapped_pos_train_triples,
                mapped_pos_test_triples=mapped_pos_test_triples,
                mapped_neg_test_triples=mapped_neg_test_triples,
                entity_to_id=self.entity_label_to_id,
                rel_to_id=self.relation_label_to_id,
                config=self.config,
                device=self.device,
                seed=self.seed,
            )
        else:  # Training Mode
            if self.is_evaluation_required:
                mapped_pos_train_triples, mapped_pos_test_triples, mapped_neg_test_triples =\
                    self._get_train_and_test_triples()
            else:
                mapped_pos_train_triples, mapped_pos_test_triples = self._get_train_triples(), None

            all_entities = np.array(list(self.entity_label_to_id.values()))

            # Initialize KG embedding model
            self.config[pkc.NUM_ENTITIES] = len(self.entity_label_to_id)
            self.config[pkc.NUM_RELATIONS] = len(self.relation_label_to_id)
            self.config[pkc.PREFERRED_DEVICE] = pkc.CPU if self.device_name == pkc.CPU else pkc.GPU
            if self.seed is not None:
                torch.manual_seed(self.seed)

            print(self.config)
            kge_model: Module = get_kge_model(config=self.config)

            batch_size = self.config[pkc.BATCH_SIZE]
            test_batch_size = self.config.get(pkc.TEST_BATCH_SIZE, batch_size)
            num_epochs = self.config[pkc.NUM_EPOCHS]
            learning_rate = self.config[pkc.LEARNING_RATE]
            neg_factor = self.config.get('neg_factor', 1)  # todo: add constants
            single_pass = self.config.get('single_pass', False)

            log.info("-------------Train KG Embeddings-------------")
            trained_model, loss_per_epoch, valloss_per_epoch = train_kge_model(
                kge_model=kge_model,
                all_entities=all_entities,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                test_batch_size=test_batch_size,
                train_triples=mapped_pos_train_triples,
                train_types=train_types,
                val_triples=mapped_pos_test_triples,
                val_types=val_types,
                device=self.device,
                neg_factor=neg_factor,
                single_pass=single_pass,
                seed=self.seed,
                model_dir=output_directory
            )

            params = self.config

            if self.is_evaluation_required:
                log.info("-------------Start Evaluation-------------")

                metric_results = compute_metric_results(
                    metrics=self.config['metrics'],
                    all_entities=all_entities,
                    kg_embedding_model=kge_model,
                    mapped_train_triples=mapped_pos_train_triples,
                    mapped_pos_test_triples=mapped_pos_test_triples,
                    mapped_neg_test_triples=mapped_neg_test_triples,
                    batch_size=test_batch_size,
                    device=self.device,
                    filter_neg_triples=self.config[pkc.FILTER_NEG_TRIPLES],
                )

            search_summary = None

        # Prepare Output
        entity_id_to_label = {
            value: key
            for key, value in self.entity_label_to_id.items()
        }
        relation_id_to_label = {
            value: key for
            key, value in self.relation_label_to_id.items()
        }
        entity_label_to_embedding = {
            entity_id_to_label[entity_id]: embedding.detach().cpu().numpy()
            for entity_id, embedding in enumerate(trained_model.entity_embeddings.weight)
        }

        if self.config[pkc.KG_EMBEDDING_MODEL_NAME] in (pkc.SE_NAME, pkc.UM_NAME):
            relation_label_to_embedding = None
        else:
            relation_label_to_embedding = {
                relation_id_to_label[relation_id]: embedding.detach().cpu().numpy()
                for relation_id, embedding in enumerate(trained_model.relation_embeddings.weight)
            }

        return _make_results(
            trained_model=trained_model,
            loss_per_epoch=loss_per_epoch,
            valloss_per_epoch=valloss_per_epoch,
            entity_to_embedding=entity_label_to_embedding,
            relation_to_embedding=relation_label_to_embedding,
            metric_results=metric_results,
            entity_to_id=self.entity_label_to_id,
            rel_to_id=self.relation_label_to_id,
            params=params,
            search_summary=search_summary
        )

    def evaluate(self, trained_model, test_path, neg_test_path,
                 metrics=[pkc.MEAN_RANK, pkc.HITS_AT_K, pkc.TRIPLE_PREDICTION]):
        # TODO: optimize run to call this function

        all_entities = np.array(list(self.entity_label_to_id.values()))

        self.config[pkc.TEST_SET_PATH] = test_path
        if pkc.TRIPLE_PREDICTION in metrics:
            self.config[pkc.NEG_TEST_PATH] = neg_test_path
        mapped_pos_train_triples, mapped_pos_test_triples, mapped_neg_test_triples =\
            self._get_train_and_test_triples()

        log.info("-------------Start Evaluation-------------")

        metric_results = compute_metric_results(
            metrics=metrics,
            all_entities=all_entities,
            kg_embedding_model=trained_model,
            mapped_train_triples=mapped_pos_train_triples,
            mapped_pos_test_triples=mapped_pos_test_triples,
            mapped_neg_test_triples=mapped_neg_test_triples,
            batch_size=self.config['test_batch_size'],
            device=self.device,
            filter_neg_triples=self.config[pkc.FILTER_NEG_TRIPLES],
        )

        # Prepare Output
        relation_id_to_label = {
            value: key for
            key, value in self.relation_label_to_id.items()
        }

        if self.config[pkc.KG_EMBEDDING_MODEL_NAME] in (pkc.SE_NAME, pkc.UM_NAME):
            relation_label_to_embedding = None
        else:
            relation_label_to_embedding = {
                relation_id_to_label[relation_id]: embedding.detach().cpu().numpy()
                for relation_id, embedding in enumerate(trained_model.relation_embeddings.weight)
            }

        return _make_results(
            trained_model=trained_model,
            loss_per_epoch=None,
            valloss_per_epoch=None,
            entity_to_embedding=None,
            relation_to_embedding=relation_label_to_embedding,
            metric_results=metric_results,
            entity_to_id=self.entity_label_to_id,
            rel_to_id=self.relation_label_to_id,
            params=self.config,
            search_summary=None
        )

    def _get_train_and_test_triples(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_pos = load_data(self.config[pkc.TRAINING_SET_PATH])
        if pkc.NEG_TEST_PATH in self.config:
            test_neg = load_data(self.config[pkc.NEG_TEST_PATH])
        else:
            test_neg = None

        if pkc.TEST_SET_PATH in self.config:
            # todo: nice handling of negative inputs
            test_pos = load_data(self.config[pkc.TEST_SET_PATH])
        else:
            train_pos, test_pos = train_test_split(
                train_pos,
                test_size=self.config.get(pkc.TEST_SET_RATIO, 0.1),
                random_state=self.seed,
            )

        return self._handle_train_and_test(train_pos, test_pos, test_neg)

    def _handle_train_and_test(self, train_pos, test_pos, test_neg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """"""
        if test_neg is not None:
            all_triples: np.ndarray = np.concatenate([train_pos, test_pos, test_neg], axis=0)
        else:
            all_triples: np.ndarray = np.concatenate([train_pos, test_pos], axis=0)
        self.entity_label_to_id, self.relation_label_to_id = create_mappings(triples=all_triples)

        #log.debug("map train")
        mapped_pos_train_triples, _, _, pos_train_types = create_mapped_triples(
            triples=train_pos,
            entity_label_to_id=self.entity_label_to_id,
            relation_label_to_id=self.relation_label_to_id,
        )

        #log.debug("map test")
        mapped_pos_test_triples, _, _, pos_test_types = create_mapped_triples(
            triples=test_pos,
            entity_label_to_id=self.entity_label_to_id,
            relation_label_to_id=self.relation_label_to_id,
        )

        if test_neg is not None:
            #log.debug("map negative test")
            mapped_neg_test_triples, _, _, _ = create_mapped_triples(
                triples=test_neg,
                entity_label_to_id=self.entity_label_to_id,
                relation_label_to_id=self.relation_label_to_id,
            )
        else:
            mapped_neg_test_triples = np.array([])

        return mapped_pos_train_triples, mapped_pos_test_triples, mapped_neg_test_triples, pos_train_types, pos_test_types

    def _get_train_triples(self):
        train_pos = load_data(self.config[pkc.TRAINING_SET_PATH])

        self.entity_label_to_id, self.relation_label_to_id = create_mappings(triples=train_pos)

        mapped_pos_train_triples, _, _, _ = create_mapped_triples(
            triples=train_pos,
            entity_label_to_id=self.entity_label_to_id,
            relation_label_to_id=self.relation_label_to_id,
        )

        return mapped_pos_train_triples


def load_data(path: Union[str, Iterable[str]]) -> np.ndarray:
    """Load data given the *path*."""
    if isinstance(path, str):
        return _load_data_helper(path)

    return np.concatenate([
        _load_data_helper(p)
        for p in path
    ])


def _load_data_helper(path: str) -> np.ndarray:
    for prefix, handler in pkc.IMPORTERS.items():
        if path.startswith(f'{prefix}:'):
            return handler(path[len(f'{prefix}:'):])

    if path.endswith('.tsv'):
        return np.reshape(np.loadtxt(
            fname=path,
            dtype=str,
            comments='@Comment@ Subject Predicate Object',
            delimiter='\t',
        ), newshape=(-1, 3))

    if path.endswith('.nt'):
        g = rdflib.Graph()
        g.parse(path, format='nt')
        return np.array(
            [
                [str(s), str(p), str(o)]
                for s, p, o in g
            ],
            dtype=np.str,
        )

    raise ValueError('''The argument to _load_data must be one of the following:

    - A string path to a .tsv file containing 3 columns corresponding to subject, predicate, and object
    - A string path to a .nt RDF file serialized in N-Triples format
    - A string NDEx network UUID prefixed by "ndex:" like in ndex:f93f402c-86d4-11e7-a10d-0ac135e8bacf
    ''')


def _make_results(
        trained_model,
        loss_per_epoch,
        valloss_per_epoch,
        entity_to_embedding: Mapping[str, np.ndarray],
        relation_to_embedding: Mapping[str, np.ndarray],
        metric_results: Optional[MetricResults],
        entity_to_id,
        rel_to_id,
        params,
        search_summary
) -> Dict:
    results = OrderedDict()
    results[pkc.TRAINED_MODEL] = trained_model
    results[pkc.LOSSES] = loss_per_epoch
    results[pkc.VAL_LOSSES] = valloss_per_epoch
    results[pkc.ENTITY_TO_EMBEDDING]: Mapping[str, np.ndarray] = entity_to_embedding
    results[pkc.RELATION_TO_EMBEDDING]: Mapping[str, np.ndarray] = relation_to_embedding
    if metric_results is not None:
        results[pkc.EVAL_SUMMARY] = {
            pkc.MEAN_RANK: metric_results.mean_rank,
            pkc.HITS_AT_K: metric_results.hits_at_k,
            "precision": metric_results.precision,  # todo: metrics names
            "recall": metric_results.recall,
            "accuracy": metric_results.accuracy,
            "f1_score": metric_results.fscore
        }
    results[pkc.ENTITY_TO_ID] = entity_to_id
    results[pkc.RELATION_TO_ID] = rel_to_id
    results[pkc.FINAL_CONFIGURATION] = params
    results['search_summary'] = search_summary
    return results
