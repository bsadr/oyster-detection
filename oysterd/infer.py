''' 
usage: python infer.py -i <model_number> -f <input_folder>
default model_number is 0
default input folder is defined in the config.py: Config.video['path'] 
'''

import argparse
from thing import Thing

parser = argparse.ArgumentParser()

parser.add_argument("-i", required=False,
                    dest="model_id",
                    default=0,
                    help="detecron2 model id (in oyster config file)")
parser.add_argument("-f", required=False,
                    dest="folder",
                    default=None,
                    help="input folder")
args = parser.parse_args()
model_id = int(args.model_id)
thing = Thing(model_id)
thing.setModel()
input_folder = args.folder
thing.infer(input_folder)