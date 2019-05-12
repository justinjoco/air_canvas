#!/usr/bin/env bash

sudo rmmod stmpe_ts
sudo modprobe stmpe_ts
sudo modprobe bcm2835-v4l2
ls /dev/video0
