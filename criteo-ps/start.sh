#!/bin/bash
sh stop.sh
sleep 1
python criteo-ps.py --ps_list="localhost:2220,localhost:2221" --task_mode="ps" --task_id=0 &
sleep 1
python criteo-ps.py --ps_list="localhost:2220,localhost:2221" --task_mode="ps" --task_id=1 &
sleep 1
python criteo-ps.py --ps_list="localhost:2220,localhost:2221" --worker_list="localhost:2230" --task_mode="worker" --task_id=0 --is_chief=True &
echo "ok"