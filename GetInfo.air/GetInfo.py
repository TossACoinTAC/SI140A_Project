# -*- encoding=utf8 -*-
__author__ = "admin"

from airtest.core.api import *

auto_setup(__file__)

from airtest.core.settings import Settings as ST

ST.LOG_DIR = r"D:\_SweetHome\WorkSpace\SI140_Project\Snapshots"

import pyautogui


def get_info():
    touch(Template(r"tpl1768634078724.png", record_pos=(-0.184, -0.24), resolution=(550, 700)))

#     touch(Template(r"tpl1768585707367.png", record_pos=(-0.005, 0.247), resolution=(550, 700)))
    touch(Template(r"tpl1768641604125.png", record_pos=(0.091, -0.182), resolution=(550, 700)))



    snapshot(filename=f"info{idx}.png", msg=f"info{idx}")
    touch(Template(r"tpl1768585836389.png", record_pos=(0.251, -0.545), resolution=(550, 700)))
   

    touch(Template(r"tpl1768641700461.png", record_pos=(0.42, -0.342), resolution=(550, 700)))




    sleep(0.1)

    for _ in range(3):
        pyautogui.scroll(-90)
    

red_packet_num=3

for idx in range(red_packet_num):
    get_info()






