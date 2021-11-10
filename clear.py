#!/usr/bin/env python
# coding=utf-8
import json
a=json.load(open('./dataset/location.json'))
print(type(a))
b=[]
for item in a:
    item.pop('cam_name')
    item.pop('cam_location')
    b.append(item)
json.dump(b,open('./dataset/loc.json','w'))
