import annotations
from config import Config

oyster_cfg = Config(0)
for d in ["train", "val"]:
    dataset = annotations.labelmeDict(oyster_cfg.folders['data'], d)
    annotations.draw(dataset)
