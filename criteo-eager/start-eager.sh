#!/bin/bash
sh stop-eager.sh
python criteo-eager.py &
echo "ok"