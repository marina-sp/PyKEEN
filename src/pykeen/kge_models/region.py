# -*- coding: utf-8 -*-

"""Implementation of the Region model."""

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.autograd
from torch import nn
import torch.nn.functional as F

from pykeen.constants import NORM_FOR_NORMALIZATION_OF_ENTITIES, SCORING_FUNCTION_NORM, REGION_NAME, RADIUS_INITIAL_VALUE
from pykeen.kge_models.base import BaseModule, slice_triples

__all__ = [
    'Region',
    'RegionConfig',
]

log = logging.getLogger(__name__)


@dataclass
class RegionConfig:
    lp_norm: str
    radius_init: float
    reg_lambda: float

    @classmethod
    def from_dict(cls, config: Dict) -> 'RegionConfig':
        """Generate an instance from a dictionary."""
        return cls(
            lp_norm=config[NORM_FOR_NORMALIZATION_OF_ENTITIES],
            scoring_function_norm=config[SCORING_FUNCTION_NORM],
            radius_init=config[RADIUS_INITIAL_VALUE],
            reg_lambda=config['reg_lambda']
        )


class Region(BaseModule):
    """A modification of TransE [borders2013]_.

     This model considers a relation as a translation from the head to a region, including the tail entity.

    .. [borders2013] Bordes, A., *et al.* (2013). `Translating embeddings for modeling multi-relational data
                     <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_
                     . NIPS.

    .. seealso::

    """

    model_name = REGION_NAME
    margin_ranking_loss_size_average: bool = True
    hyper_params = BaseModule.hyper_params + [
        NORM_FOR_NORMALIZATION_OF_ENTITIES,
        RADIUS_INITIAL_VALUE
    ]

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        config = RegionConfig.from_dict(config)

        # Embeddings
        self.l_p_norm_entities = config.lp_norm
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        self.relation_regions = nn.Embedding(self.num_relations, 1)
        self.reg_l = config.reg_lambda
        self.init_radius = config.radius_init

        # TODO: add config parameter and move to base class
        self.criterion = nn.MarginRankingLoss(
            margin=self.margin_loss,
            size_average=self.margin_ranking_loss_size_average
        )

        # Output type (used for scoring)
        self.prob_mode = True

        self._initialize()

    def _initialize(self):
        embeddings_init_bound = 6 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )

        # todo: relation initialization
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-embeddings_init_bound,
            b=+embeddings_init_bound,
        )

        # todo: relation initialization
        nn.init.constant_(
            self.relation_regions.weight.data,
            self.init_radius
        )

        # todo: relation normalization at init
        norms = torch.norm(self.relation_embeddings.weight, p=2, dim=1).data
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.div(
            norms.view(self.num_relations, 1).expand_as(self.relation_embeddings.weight))

    def predict(self, triples):
        scores = self._score_triples(triples)
        return scores.detach().cpu().numpy()

    def forward(self, batch_positives, batch_negatives):
        # Normalize embeddings of entities
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))
        # TODO: what is going on

        positive_scores = - torch.log(self._score_triples(batch_positives))
        negative_scores = - torch.log(1 - self._score_triples(batch_negatives))
        loss = self._compute_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss

    def _compute_loss(self, positive_scores, negative_scores):
        y = np.repeat([-1], repeats=positive_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        # Scores for the positive and negative triples
        positive_scores = torch.tensor(positive_scores, dtype=torch.float, device=self.device)
        negative_scores = torch.tensor(negative_scores, dtype=torch.float, device=self.device)

        loss = self.criterion(positive_scores, - negative_scores, y)
        #print("loss: %0.2f"%loss)
        # TODO: regularization once or for every element
        # todo: if here -- then once a batch -- normalized to batch size?
        loss = loss.add(
            self.reg_l *
            torch.pow(
                torch.norm(self.relation_regions.weight.data.view(-1), 2),
                2))
        #print("reg loss: %0.2f"%loss)
        #print(positive_scores, negative_scores, loss, loss_)
        return loss

    def _score_triples(self, triples):
        head_embeddings, relation_embeddings, tail_embeddings, relation_regions = \
            self._get_triple_embeddings(triples)
        scores = self._compute_scores(
            head_embeddings,
            relation_embeddings,
            tail_embeddings,
            relation_regions)
        return scores

    def _compute_scores(self, head_embeddings, relation_embeddings, tail_embeddings, regions):
        """Compute the scores based on the head, relation, and tail embeddings.

        :param head_embeddings: embeddings of head entities of dimension batchsize x embedding_dim
        :param relation_embeddings: embeddings of relation embeddings of dimension batchsize x embedding_dim
        :param tail_embeddings: embeddings of tail entities of dimension batchsize x embedding_dim
        :param regions: relation region descriptor of dimension batchsize x 1
        :return: Tensor of dimension batch_size containing the scores for each batch element
        """
        m_x    = (head_embeddings + relation_embeddings - tail_embeddings).unsqueeze(-1)
        sigmas = torch.log(1 + torch.exp(regions)).unsqueeze(-1).unsqueeze(-1)
        region_m = (sigmas * torch.eye(self.embedding_dim, device=self.device)).squeeze(1)
        dists  = torch.matmul(
            torch.matmul(m_x.transpose(-1, -2), region_m),
            m_x).squeeze(-1)
        # TODO: try other activation, like tanh instead of sigmoid
        if (dists == 0).any():
            print("zero distances in loss computation")
        probs = 1.0 / (1 + dists)
        #print("Shapes", m_x.shape, sigma, A.shape, dists.shape, probs.shape)
        #print("Dist values: ", dists[:10])
        #print("Probs values: ", probs[:10])
        #print("M x: ", m_x)
        return probs

    def _get_triple_embeddings(self, triples):
        heads, relations, tails = slice_triples(triples)
        return (
            self._get_entity_embeddings(heads),
            self._get_relation_embeddings(relations),
            self._get_entity_embeddings(tails),
            self._get_relation_regions(relations)
        )

    def _get_relation_embeddings(self, relations):
        return self.relation_embeddings(relations).view(-1, self.embedding_dim)

    def _get_relation_regions(self, relations):
        return self.relation_regions(relations).view(-1, 1)