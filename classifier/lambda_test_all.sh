#!/bin/bash

EPOCHS=10

python test.py --num_epochs $EPOCHS --study layer
python test.py --num_epochs $EPOCHS --study qubit