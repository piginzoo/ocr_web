#!/usr/bin/env bash

Date=$(date +%Y%m%d%H%M)

imgn=$1

echo "TFS api test demo"

python -m server.tfs_api_demo \
    --imgn=$imgn