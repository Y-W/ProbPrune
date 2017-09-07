#!/bin/sh

cd "$(dirname "$(dirname "$(readlink -fm "$0")")")"

PYTHONPATH=src python -m ProbPrune.train_vanilla $1
