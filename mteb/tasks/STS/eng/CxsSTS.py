from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS

import datasets

class CxsSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="CXS-STS",
        dataset={
            "path": "",
            "revision": "",
        },
        description="cxs similarity test",
        reference="https://example",
        type="STS",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="",
        descriptive_stats={"n_samples": None, "avg_character_length": None},
    )

    def load_data(self, **kwargs):
        print("loading data")
        if self.data_loaded:
            return
        self.dataset =  datasets.load_dataset("json", data_files={"test": "localization.json"})
        
        self.dataset_transform()
        self.data_loaded = True
    
    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
