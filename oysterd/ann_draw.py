import annotations
from config import Config
from detectron2.data import DatasetCatalog, MetadataCatalog


oyster_cfg = Config(0)
for d in ["train", "val"]:
    dataset = annotations.labelmeDict(oyster_cfg.folders['data'], d)
    # annotations.draw(dataset)

    DatasetCatalog.register("oyster_" + d, lambda d=d: annotations.labelmeDict(oyster_cfg.folders['data'], d))
    MetadataCatalog.get("oyster_" + d).set(thing_classes=["oyster"])
    oyster_metadata = MetadataCatalog.get("oyster_" + d)
    annotations.draw_masks(dataset, oyster_metadata)
