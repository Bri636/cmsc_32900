#!/bin/bash
EPOCHS=15

# for layer study
# python generate_plots.py --study layer --type train --epochs $EPOCHS
# python generate_plots.py --study layer --type test --epochs $EPOCHS

# for qubit study
python generate_plots.py --study qubit --type train --epochs $EPOCHS
python generate_plots.py --study qubit --type test --epochs $EPOCHS