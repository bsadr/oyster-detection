import argparse
from thing import Thing

parser = argparse.ArgumentParser()

parser.add_argument("-i", required=False,
                    dest="cfg_id",
                    default=0,
                    help="detecron2 model id (in oyster config file)")
parser.add_argument("-p", required=False,
                    dest="predict",
                    default=True,
                    help="whether or not predict the val images")
args = parser.parse_args()
cfg_id = int(args.cfg_id)
oyster = Thing(cfg_id)
oyster.register()
oyster.train()
oyster.evaluate()

if args.predict:
    oyster.infer()
