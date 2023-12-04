#!/bin/bash

docker build --tag 'tab' .
echo "--------------------------------"
docker image ls
echo "--------------------------------"
docker ps -a
echo "--------------------------------"
# docker-compose up -d
# docker exec -it telegram_bot_tab_1 /bin/bash

