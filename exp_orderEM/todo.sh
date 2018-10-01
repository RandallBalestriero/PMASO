#!/bin/bash

#Cosmetics
DIV=$(echo "------------------------------------------")

#Begin normal stuff
echo ${DIV}
echo "Looking through all processes"
ps aux | head -n 1
ps aux | grep $1 | grep -v grep | grep -v find_who

#Begin docker stuff
echo ${DIV}
echo "Trying docker images"

CONS=$(docker ps | awk 'BEGIN { FS=" "; }{ print $1; }' | grep -v CONTAINER)

PID=$1

for X in ${CONS}
do
        ID=$(docker top ${X} | grep ${PID})
        if [ -n "$(echo ${ID})" ]
                then
                echo ${DIV}
                echo "Container ID: " ${X}
                echo ""
#               echo $(docker top ${X} | head -n 1)
#               echo ${ID}
                docker ps | head -n 1
                docker ps | grep ${X}
        fi
done

