#!/bin/sh

cd "$(dirname "$(dirname "$(readlink -fm "$0")")")"

rsync -aP ./models yw@home:/home/yw/research/ProbPrune/
