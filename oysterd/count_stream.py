import argparse
from thing import Thing
from stream import Stream

parser = argparse.ArgumentParser()
parser.add_argument("-i", required=False,
                    dest="model_id",
                    default=0,
                    help="detecron2 model id (in oyster config file)")
parser.add_argument("-p", required=False,
                    dest="path",
                    default=None,
                    help="video path")
parser.add_argument("-fps", required=False,
                    dest="fps",
                    default=-1,
                    help="fps")
parser.add_argument("-fs", required=False,
                    dest="fs",
                    default=1,
                    help="start frame")
parser.add_argument("-fe", required=False,
                    dest="fe",
                    default=-1,
                    help="end frame, -1 for the last")
parser.add_argument("-t", required=False,
                    dest="tmp",
                    default=None,
                    help="temp path")
args = parser.parse_args()

model_id = int(args.model_id)
thing = Thing(model_id)
vdata = dict (
    path = args.path if args.path else thing.cfg_thing.video["path"],
    fps = int(args.fps) if args.fps else thing.cfg_thing.video["fps"],
    fs = int(args.fs) if args.fs else thing.cfg_thing.video["fs"],
    fe = int(args.fe) if args.fe else thing.cfg_thing.video["fe"],
    tmp = args.tmp if args.tmp else thing.cfg_thing.video["tmp"]
)
stream = Stream(thing, vdata)
#stream.countVideo()
#stream.extractVideo()
stream.lastVideo()
