#!/bin/bash

# 对代码进行格式化, 前提是需要安装下black

cd /data/projects/unitree_go2_fixdim
black --line-length 120 *
