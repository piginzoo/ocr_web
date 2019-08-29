#!/usr/bin/env bash

Date=$(date +%Y%m%d%H%M)

imgn=$1
IP=$2
echo "TFS api test demo"

nohup python -m server.tfs_api_demo \
    --imgn=$imgn --IP=$2 \
    >> ./logs/tfs_api_demo_$Date.log 2>&1 &
