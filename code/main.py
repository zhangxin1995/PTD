#!/usr/bin/env python
# coding=utf-8
import argparse
from ruamel.yaml import YAML
from module.local_crf_main import local_crf_main
dict2main={'local_CRF':local_crf_main}


def build_main(name):
    return dict2main[name]
    
    
yaml = YAML(typ='safe')
def load_yaml(p):
    with open(p) as infile:
        data=yaml.load(infile)
    return data
parser=argparse.ArgumentParser()
parser.add_argument("--yaml",help="echo the string")
args=parser.parse_args()

config=load_yaml(args.yaml)
print(config)
main=build_main(config['method'])
main(config)


