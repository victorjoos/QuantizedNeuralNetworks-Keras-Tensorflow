#!/usr/bin/env bash
# cuDNN

GPU=$1

CONFIG=config
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
LOGDIR="$TIME/logs"
PARAMS="date=${TIME} nres=3 pfilt=1 cuda=${GPU}"
mkdir "$TIME"
mkdir "$TIME/logs"
echo $PARAMS > "$LOGDIR/config.txt"

# standard float
python3 train.py -c "$CONFIG" -o $PARAMS network_type='float' fold=1 lr=0.01 | tee "$LOGDIR/ff_fold1.out"
python3 train.py -c "$CONFIG" -o $PARAMS network_type='float' fold=2 lr=0.01 | tee "$LOGDIR/ff_fold2.out"
python3 train.py -c "$CONFIG" -o $PARAMS network_type='float' fold=3 lr=0.01 | tee "$LOGDIR/ff_fold3.out"
python3 train.py -c "$CONFIG" -o $PARAMS network_type='float' fold=4 lr=0.01 | tee "$LOGDIR/ff_fold4.out"
python3 train.py -c "$CONFIG" -o $PARAMS network_type='float' fold=5 lr=0.01 | tee "$LOGDIR/ff_fold5.out"
python3 train.py -c "$CONFIG" -o $PARAMS network_type='float' fold=6 lr=0.01 | tee "$LOGDIR/ff_fold6.out"
python3 train.py -c "$CONFIG" -o $PARAMS network_type='float' fold=7 lr=0.01 | tee "$LOGDIR/ff_fold7.out"
python3 train.py -c "$CONFIG" -o $PARAMS network_type='float' fold=8 lr=0.01 | tee "$LOGDIR/ff_fold8.out"
python3 train.py -c "$CONFIG" -o $PARAMS network_type='float' fold=9 lr=0.01 | tee "$LOGDIR/ff_fold9.out"
python3 train.py -c "$CONFIG" -o $PARAMS network_type='float' fold=0 lr=0.01 | tee "$LOGDIR/ff_fold0.out"


# python3 train.py -c "$CONFIG" -o $PARAMS network_type='qtnn' abits=4 fold=1 lr=0.01 | tee "$LOGDIR/t4_fold1.out"
# python3 train.py -c "$CONFIG" -o $PARAMS network_type='qtnn' abits=4 fold=2 lr=0.01 | tee "$LOGDIR/t4_fold2.out"
# python3 train.py -c "$CONFIG" -o $PARAMS network_type='qtnn' abits=4 fold=3 lr=0.01 | tee "$LOGDIR/t4_fold3.out"
# python3 train.py -c "$CONFIG" -o $PARAMS network_type='qtnn' abits=4 fold=4 lr=0.01 | tee "$LOGDIR/t4_fold4.out"
# python3 train.py -c "$CONFIG" -o $PARAMS network_type='qtnn' abits=4 fold=5 lr=0.01 | tee "$LOGDIR/t4_fold5.out"
# python3 train.py -c "$CONFIG" -o $PARAMS network_type='qtnn' abits=4 fold=6 lr=0.01 | tee "$LOGDIR/t4_fold6.out"
# python3 train.py -c "$CONFIG" -o $PARAMS network_type='qtnn' abits=4 fold=7 lr=0.01 | tee "$LOGDIR/t4_fold7.out"
# python3 train.py -c "$CONFIG" -o $PARAMS network_type='qtnn' abits=4 fold=8 lr=0.01 | tee "$LOGDIR/t4_fold8.out"
# python3 train.py -c "$CONFIG" -o $PARAMS network_type='qtnn' abits=4 fold=9 lr=0.01 | tee "$LOGDIR/t4_fold9.out"
# python3 train.py -c "$CONFIG" -o $PARAMS network_type='qtnn' abits=4 fold=0 lr=0.01 | tee "$LOGDIR/t4_fold0.out"
