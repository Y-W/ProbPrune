#!/bin/sh

cd "$(dirname "$(dirname "$(readlink -fm "$0")")")"

rsync -aP yw@home:/home/yw/research/ProbPrune/models ./
