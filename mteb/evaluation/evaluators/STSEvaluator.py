from __future__ import annotations

import logging
import json  # Add JSON for exporting the results
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from mteb.encoder_interface import Encoder, EncoderWithSimilarity
from .Evaluator import Evaluator
from .model_encode import model_encode

logger = logging.getLogger(__name__)

class STSEvaluator(Evaluator):
    def __init__(
        self,
        sentences1,
        sentences2,
        gold_scores,
        task_name: str | None = None,
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit is not None:
            sentences1 = sentences1[:limit]
            sentences2 = sentences2[:limit]
            gold_scores = gold_scores[:limit]
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.gold_scores = gold_scores
        self.task_name = task_name

    def __call__(
        self,
        model: Encoder | EncoderWithSimilarity,
        *,
        encode_kwargs: dict[str, Any] = {},
    ):
        embeddings1 = model_encode(
            self.sentences1, model=model, task_name=self.task_name, **encode_kwargs
        )
        embeddings2 = model_encode(
            self.sentences2, model=model, task_name=self.task_name, **encode_kwargs
        )

        logger.info("Evaluating...")
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)

        cosine_pearson, _ = pearsonr(self.gold_scores, cosine_scores)
        cosine_spearman, _ = spearmanr(self.gold_scores, cosine_scores)

        manhatten_pearson, _ = pearsonr(self.gold_scores, manhattan_distances)
        manhatten_spearman, _ = spearmanr(self.gold_scores, manhattan_distances)

        euclidean_pearson, _ = pearsonr(self.gold_scores, euclidean_distances)
        euclidean_spearman, _ = spearmanr(self.gold_scores, euclidean_distances)

        similarity_scores = None
        if hasattr(model, "similarity_pairwise"):
            similarity_scores = model.similarity_pairwise(embeddings1, embeddings2) 
        elif hasattr(model, "similarity"):
            _similarity_scores = [
                float(model.similarity(e1, e2)) 
                for e1, e2 in zip(embeddings1, embeddings2)
            ]
            similarity_scores = np.array(_similarity_scores)

        if similarity_scores is not None:
            pearson, _ = pearsonr(self.gold_scores, similarity_scores)
            spearman, _ = spearmanr(self.gold_scores, similarity_scores)
        else:
            pearson = cosine_pearson
            spearman = cosine_spearman

        print("Cosine Pearson:", cosine_pearson)
        print(type(cosine_pearson))
        entries = []
        for s1, s2, emb1, emb2, cos_sim, gold_score in zip(
            self.sentences1, self.sentences2, embeddings1, embeddings2, cosine_scores, self.gold_scores
        ):
            entry = {
                "sentence1": s1,
                "sentence2": s2,
                "embedding1": emb1.tolist(),  
                "embedding2": emb2.tolist(),  
                "cos_similarity": float(cos_sim),  
                "score": float(gold_score), 
            }
            entries.append(entry)

        # Save the data to a JSON file
        output_filename = 'embeddings_and_similarities_desc.json'
        with open(output_filename, 'w') as f:
            json.dump(entries, f, indent=4)
        logger.info(f"Results saved to {output_filename}")

        return {
            "pearson": pearson,
            "spearman": spearman,
            "cosine_pearson": cosine_pearson,
            "cosine_spearman": cosine_spearman,
            "manhattan_pearson": manhatten_pearson,
            "manhattan_spearman": manhatten_spearman,
            "euclidean_pearson": euclidean_pearson,
            "euclidean_spearman": euclidean_spearman,
        }
