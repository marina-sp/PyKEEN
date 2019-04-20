# -*- coding: utf-8 -*-

"""Script to compute mean rank and hits@k."""

import logging
import timeit
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)

DEFAULT_HITS_AT_K = [1, 3, 5, 10]


def _hash_triples(triples: Iterable[Hashable]) -> int:
    """Hash a list of triples."""
    return hash(tuple(triples))


def update_hits_at_k(
        hits_at_k_values: Dict[int, List[float]],
        rank_of_positive_subject_based: int,
        rank_of_positive_object_based: int
) -> None:
    """Update the Hits@K dictionary for two values."""
    for k, values in hits_at_k_values.items():
        if rank_of_positive_subject_based < k:
            values.append(1.0)
        else:
            values.append(0.0)

        if rank_of_positive_object_based < k:
            values.append(1.0)
        else:
            values.append(0.0)


def _create_corrupted_triples(triple, all_entities, device):
    candidate_entities_subject_based = np.delete(arr=all_entities, obj=triple[0:1])
    candidate_entities_subject_based = np.reshape(candidate_entities_subject_based, newshape=(-1, 1))
    candidate_entities_object_based = np.delete(arr=all_entities, obj=triple[2:3])
    candidate_entities_object_based = np.reshape(candidate_entities_object_based, newshape=(-1, 1))

    # Extract current test tuple: Either (subject,predicate) or (predicate,object)
    tuple_subject_based = np.reshape(a=triple[1:3], newshape=(1, 2))
    tuple_object_based = np.reshape(a=triple[0:2], newshape=(1, 2))

    # Copy current test tuple
    tuples_subject_based = np.repeat(a=tuple_subject_based, repeats=candidate_entities_subject_based.shape[0],
                                     axis=0)
    tuples_object_based = np.repeat(a=tuple_object_based, repeats=candidate_entities_object_based.shape[0], axis=0)

    corrupted_subject_based = np.concatenate([candidate_entities_subject_based, tuples_subject_based], axis=1)
    corrupted_subject_based = torch.tensor(corrupted_subject_based, dtype=torch.long, device=device)

    corrupted_object_based = np.concatenate([tuples_object_based, candidate_entities_object_based], axis=1)
    corrupted_object_based = torch.tensor(corrupted_object_based, dtype=torch.long, device=device)

    return corrupted_subject_based, corrupted_object_based


def _filter_corrupted_triples(
        corrupted_subject_based,
        corrupted_object_based,
        all_pos_triples_hashed,
):
    # TODO: Check
    corrupted_subject_based_hashed = np.apply_along_axis(_hash_triples, 1, corrupted_subject_based)
    mask = np.in1d(corrupted_subject_based_hashed, all_pos_triples_hashed, invert=True)
    mask = np.where(mask)[0]
    corrupted_subject_based = corrupted_subject_based[mask]

    corrupted_object_based_hashed = np.apply_along_axis(_hash_triples, 1, corrupted_object_based)
    mask = np.in1d(corrupted_object_based_hashed, all_pos_triples_hashed, invert=True)
    mask = np.where(mask)[0]

    if mask.size == 0:
        raise Exception("User selected filtered metric computation, but all corrupted triples exists"
                        "also as positive triples.")
    corrupted_object_based = corrupted_object_based[mask]

    return corrupted_subject_based, corrupted_object_based


def _compute_filtered_rank(
        kg_embedding_model,
        score_of_positive,
        corrupted_subject_based,
        corrupted_object_based,
        batch_size,
        device,
        all_pos_triples_hashed,
) -> Tuple[int, int]:
    """

    :param kg_embedding_model:
    :param score_of_positive:
    :param corrupted_subject_based:
    :param corrupted_object_based:
    :param device:
    :param all_pos_triples_hashed:
    """
    corrupted_subject_based, corrupted_object_based = _filter_corrupted_triples(
        corrupted_subject_based=corrupted_subject_based,
        corrupted_object_based=corrupted_object_based,
        all_pos_triples_hashed=all_pos_triples_hashed)

    return _compute_rank(
        kg_embedding_model=kg_embedding_model,
        score_of_positive=score_of_positive,
        corrupted_subject_based=corrupted_subject_based,
        corrupted_object_based=corrupted_object_based,
        batch_size=batch_size,
        device=device,
        all_pos_triples_hashed=all_pos_triples_hashed,
    )


def _compute_rank(
        kg_embedding_model,
        score_of_positive,
        corrupted_subject_based,
        corrupted_object_based,
        batch_size,
        device,
        all_pos_triples_hashed=None,
) -> Tuple[int, int]:
    """

    :param kg_embedding_model:
    :param score_of_positive:
    :param corrupted_subject_based:
    :param corrupted_object_based:
    :param device:
    :param all_pos_triples_hashed: This parameter isn't used but is necessary for compatability
    """
    corrupted_scores = np.ndarray((len(corrupted_subject_based)+len(corrupted_object_based),1))
    #print("num of corrupted: ", len(corrupted_scores), corrupted_scores)
    for i, batch in enumerate(_split_list_in_batches(
            torch.cat([corrupted_subject_based, corrupted_object_based], dim=0),
            batch_size)):
        corrupted_scores[i * batch_size: (i + 1) * batch_size] = kg_embedding_model.predict(batch)

    scores_of_corrupted_subjects = corrupted_scores[:len(corrupted_subject_based)]
    scores_of_corrupted_objects = corrupted_scores[len(corrupted_subject_based):]

    scores_subject_based = np.append(arr=scores_of_corrupted_subjects, values=score_of_positive)
    index_of_pos_subject_based = scores_subject_based.size - 1

    scores_object_based = np.append(arr=scores_of_corrupted_objects, values=score_of_positive)
    index_of_pos_object_based = scores_object_based.size - 1



    _, sorted_score_indices_subject_based = torch.sort(torch.tensor(scores_subject_based, dtype=torch.float),
                                                       descending=kg_embedding_model.prob_mode)
    sorted_score_indices_subject_based = sorted_score_indices_subject_based.cpu().numpy()

    _, sorted_score_indices_object_based = torch.sort(torch.tensor(scores_object_based, dtype=torch.float),
                                                      descending=kg_embedding_model.prob_mode)
    sorted_score_indices_object_based = sorted_score_indices_object_based.cpu().numpy()

    # Get index of first occurrence that fulfills the condition
    rank_of_positive_subject_based = np.where(sorted_score_indices_subject_based == index_of_pos_subject_based)[0][0]
    rank_of_positive_object_based = np.where(sorted_score_indices_object_based == index_of_pos_object_based)[0][0]

    return (
        rank_of_positive_subject_based,
        rank_of_positive_object_based,
    )

def _split_list_in_batches(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

@dataclass
class MetricResults:
    """Results from computing metrics."""

    mean_rank: float
    hits_at_k: Dict[int, float]


def compute_metric_results(
        all_entities,
        kg_embedding_model,
        mapped_train_triples,
        mapped_test_triples,
        batch_size,
        device,
        filter_neg_triples=False,
        ks: Optional[List[int]] = None
) -> MetricResults:
    """Compute the metric results.

    :param all_entities:
    :param kg_embedding_model:
    :param mapped_train_triples:
    :param mapped_test_triples:
    :param device:
    :param filter_neg_triples:
    :param ks:
    :return:
    """
    start = timeit.default_timer()

    ranks: List[int] = []
    hits_at_k_values = {
        k: []
        for k in (ks or DEFAULT_HITS_AT_K)
    }
    kg_embedding_model = kg_embedding_model.eval()
    kg_embedding_model = kg_embedding_model.to(device)

    all_pos_triples = np.concatenate([mapped_train_triples, mapped_test_triples], axis=0)
    all_pos_triples_hashed = np.apply_along_axis(_hash_triples, 1, all_pos_triples)

    compute_rank_fct: Callable[..., Tuple[int, int]] = (
        _compute_filtered_rank
        if filter_neg_triples else
        _compute_rank
    )

    all_scores = np.ndarray((len(mapped_test_triples),1))

    all_triples = torch.tensor(mapped_test_triples, dtype=torch.long, device=device)

    # predict all triples
    for i, batch in enumerate(_split_list_in_batches(all_triples, batch_size)):
        predictions = kg_embedding_model.predict(batch)
        all_scores[i*batch_size: (i+1)*batch_size] = predictions

    # Corrupt triples
    for i, pos_triple in enumerate(tqdm(mapped_test_triples)):
        corrupted_subject_based, corrupted_object_based = _create_corrupted_triples(
            triple=pos_triple,
            all_entities=all_entities,
            device=device,
        )

        rank_of_positive_subject_based, rank_of_positive_object_based = compute_rank_fct(
            kg_embedding_model=kg_embedding_model,
            score_of_positive=all_scores[i],
            corrupted_subject_based=corrupted_subject_based,
            corrupted_object_based=corrupted_object_based,
            batch_size=batch_size,
            device=device,
            all_pos_triples_hashed=all_pos_triples_hashed,
        )

        ranks.append(rank_of_positive_subject_based)
        ranks.append(rank_of_positive_object_based)

        # Compute hits@k for k in {1,3,5,10}
        update_hits_at_k(
            hits_at_k_values,
            rank_of_positive_subject_based=rank_of_positive_subject_based,
            rank_of_positive_object_based=rank_of_positive_object_based,
        )

    mean_rank = float(np.mean(ranks))
    hits_at_k: Dict[int, float] = {
        k: np.mean(values)
        for k, values in hits_at_k_values.items()
    }

    stop = timeit.default_timer()
    log.info("Evaluation took %.2fs seconds", stop - start)

    return MetricResults(
        mean_rank=mean_rank,
        hits_at_k=hits_at_k,
    )
