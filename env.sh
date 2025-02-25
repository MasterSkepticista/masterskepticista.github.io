#!/bin/bash
set -Eeuo pipefail
source ~/set_proxy.sh
export GIT_CONFIG_GLOBAL=
git config --local user.name "MasterSkepticista"
git config --local user.email "kbshah1998@outlook.com"
echo "Done"
