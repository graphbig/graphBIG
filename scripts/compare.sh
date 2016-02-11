#!/bin/bash

if [ "$#" -ne 3 ];
then
    echo "Usage: $0 <output> <expected> <diff>"
    exit
fi

OUTPUT_FILE=$1
EXPECTED_OUTPUT_FILE=$2
DIFF_FILE=$3

diff ${OUTPUT_FILE} ${EXPECTED_OUTPUT_FILE} &> ${DIFF_FILE}

echo "****************************************************"
if [ -s ${DIFF_FILE} ]; 
then
	echo "TEST FAILED: please check ${DIFF_FILE} for details"
    echo "****************************************************"
    if [ "$TRAVIS" == "on" ];
    then
        cat ${DIFF_FILE}
    fi
    exit 1
else
	echo "TEST PASSED"
	rm -rf ${DIFF_FILE}
    echo "****************************************************"
    exit 0
fi

