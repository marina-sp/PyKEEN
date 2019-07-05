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

from pykeen.constants import NORM_FOR_NORMALIZATION_OF_ENTITIES, REGION_NAME, RADIUS_INITIAL_VALUE
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
    loss_type: str
    neg_factor: float
    region_type: str

    @classmethod
    def from_dict(cls, config: Dict) -> 'RegionConfig':
        """Generate an instance from a dictionary."""
        return cls(
            lp_norm=config[NORM_FOR_NORMALIZATION_OF_ENTITIES],
            radius_init=config[RADIUS_INITIAL_VALUE],
            reg_lambda=config['reg_lambda'],
            loss_type=config.get('loss_type', 'NLL'),
            neg_factor=config.get('neg_factor', 1),
            region_type=config.get('region_type', 'sphere')
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
    margin_ranking_loss_size_average: bool = False
    hyper_params = BaseModule.hyper_params + [
        NORM_FOR_NORMALIZATION_OF_ENTITIES,
        RADIUS_INITIAL_VALUE,
        'reg_lambda',
        'loss_type',
        'neg_factor',
        'region_type'
    ]
    single_threshold = False

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        config = RegionConfig.from_dict(config)

        # Embeddings
        self.l_p_norm_entities = config.lp_norm
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        if config.region_type == 'sphere':
            self.region_dim = 1
            self.relation_regions = nn.Embedding(
                self.num_relations,
                1)
        elif config.region_type == 'ellipse':
            self.region_dim = 2
            self.relation_regions = nn.Embedding(
                self.num_relations,
                self.embedding_dim)
        else:
            self.region_dim = 3
            self.relation_regions = nn.Embedding(
                self.num_relations,
                self.embedding_dim * (self.embedding_dim - 1) // 2 + self.embedding_dim)

        self.reg_l = config.reg_lambda
        self.init_radius = config.radius_init

        # TODO: add config parameter and move to base class
        self.loss_type = config.loss_type
        if config.loss_type == 'MRL':
            self.criterion = nn.MarginRankingLoss(
                margin=self.margin_loss,
                size_average= False #  self.margin_ranking_loss_size_average
            )
            self.single_pass = False
            self.forward = self._forward_split
        elif config.loss_type == 'NLL':
            self.criterion = nn.NLLLoss(
                size_average= False #  self.margin_ranking_loss_size_average
            )  # todo: add weights for pos and neg classes
            self.margin_loss = 0
            self.forward = self._forward_single
            self.single_pass = True

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

    def _forward_split(self, batch_positives, batch_negatives):
        # Normalize embeddings of entities
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))

        pos = self._score_triples(batch_positives)
        positive_scores = - torch.log(pos)

        neg = self._score_triples(batch_negatives)
        if (pos < 0).any() or (neg < 0).any():
            log.debug("negative input to log function")
        if (pos == 0).any():
            log.debug("zero input from pos to log function")
        if (neg == 0).any():
            log.debug("zero input from neg to log function")
        negative_scores = - torch.log(neg)
        loss = self._compute_split_loss(positive_scores=positive_scores, negative_scores=negative_scores)
        return loss

    def _compute_split_loss(self, positive_scores, negative_scores):
        y = np.repeat([-1], repeats=positive_scores.shape[0])
        y = torch.tensor(y, dtype=torch.float, device=self.device)

        loss = torch.sum(
            torch.max(positive_scores - negative_scores + self.margin_loss, torch.tensor(0.0, device=self.device))
        )
        #loss = self.criterion(positive_scores, negative_scores, y)
        #print("loss: %0.2f"%loss)
        if self.reg_l:
            embeddings = self.relation_regions(torch.tensor(range(self.num_relations), device=self.device))
            reg_loss = self.reg_l * len(positive_scores) / self.embedding_dim / self.num_relations * \
                torch.pow(
                    torch.norm(embeddings.reshape(-1), 2),
                    2)
            loss = loss + reg_loss
        #print("reg loss: %0.2f"%reg_loss)
        #print(positive_scores, negative_scores, loss, loss_)

        return loss

    def _forward_single(self, batch, targets):
        # Normalize embeddings of entities
        norms = torch.norm(self.entity_embeddings.weight, p=self.l_p_norm_entities, dim=1).data
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.div(
            norms.view(self.num_entities, 1).expand_as(self.entity_embeddings.weight))

        scores = self._score_triples(batch)
        loss = self._compute_single_loss(scores, targets)
        return loss

    def _compute_single_loss(self, scores_1d, targets):
        pos_mask = torch.tensor((targets == 1), dtype=torch.float, device=self.device).unsqueeze(1)
        scores_pos = scores_1d * pos_mask + (1 - scores_1d) * (1 - pos_mask)
        targets = torch.tensor(np.zeros(targets.shape[0]), dtype=torch.long, device=self.device)
        loss = self.criterion(torch.log(scores_pos), targets)

        # TODO: regularization once or for every element
        # todo: if here -- then once a batch -- normalized to batch size?
        if self.reg_l:
            embeddings = self.relation_regions(torch.tensor(range(self.num_relations), device=self.device))
            loss = loss.add(
                self.reg_l * len(targets) / self.embedding_dim / self.num_relations *
                torch.pow(
                    torch.norm(embeddings.view(-1), 2),
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

    def _compute_scores(self, head_embeddings, relation_embeddings, tail_embeddings, region_m):
        """Compute the scores based on the head, relation, and tail embeddings.

        :param head_embeddings: embeddings of head entities of dimension batchsize x embedding_dim
        :param relation_embeddings: embeddings of relation embeddings of dimension batchsize x embedding_dim
        :param tail_embeddings: embeddings of tail entities of dimension batchsize x embedding_dim
        :param regions: relation region descriptor of dimension batchsize x 1
        :return: Tensor of dimension batch_size containing the scores for each batch element
        """
        m_x    = (head_embeddings + relation_embeddings - tail_embeddings).unsqueeze(-1)
        dists  = torch.matmul(
            torch.matmul(m_x.transpose(-1, -2), region_m),
            m_x).squeeze(-1)
        # TODO: try other activation, like tanh instead of sigmoid
        if (dists == 0).any():
            print("zero distances in loss computation")
        probs = 1.0 / (1 + dists + 1e-15)
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
        if self.region_dim != 3:
            if self.region_dim == 1:
                values = self.relation_regions(relations).view(-1,1,1)
            else:
                values = self.relation_regions(relations).view(-1, self.embedding_dim, 1)
            sigmas = torch.log(1 + torch.exp(values))
            matrix = (sigmas * torch.eye(self.embedding_dim, device=self.device))
        else:
            bs = len(relations)
            emb = self.embedding_dim

            # get embedding values
            values = self.relation_regions(relations)

            # fill triangular matrix
            mask = torch.ones(emb, emb, device=self.device).tril() == 1
            triang_index = torch.stack([mask for _ in range(bs)])
            triang_matrix = torch.zeros(bs, emb, emb, device=self.device)
            triang_matrix[triang_index] = values.view(-1)

            # positive diagonal
            mask = torch.eye(emb, emb, device=self.device) == 1
            diag_index = torch.stack([mask for _ in range(bs)])
            triang_matrix[diag_index] = torch.log(1 + torch.exp(triang_matrix[diag_index]))

            # get region matrix
            matrix = triang_matrix.bmm(triang_matrix.transpose(-2, -1))

        return matrix
