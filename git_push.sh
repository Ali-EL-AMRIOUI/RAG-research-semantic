#!/bin/bash

git add .
read -p "Commit message: " msg
if [ -z "$msg" ]
then
    msg="Update $(date +'%Y-%m-%d %H:%M:%S')"
fi
git commit -m "$msg"
git push origin main