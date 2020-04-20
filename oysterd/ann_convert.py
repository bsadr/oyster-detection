import annotations
from config import Config

oyster_cfg = Config(0)
for d in ["train", "val"]:
    annotations.exp_makesense_labelme(oyster_cfg.folders['makesense'], d)
