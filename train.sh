#!/usr/bin/env bash
# cuDNN

NRES=$1
LR=$2
GPU=$3
CONFIG=config
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
LOGDIR="$TIME/logs"
mkdir "$TIME"
mkdir "$TIME/logs"
# standard float
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='float' | tee "$LOGDIR/ff.out"

# binary
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='full-bnn' | tee "$LOGDIR/bb.out"
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='qbnn' abits=2 | tee "$LOGDIR/b2.out"
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='qbnn' abits=4 | tee "$LOGDIR/b4.out"
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='qbnn' abits=8 | tee "$LOGDIR/b8.out"
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='bnn' abits=8 | tee "$LOGDIR/bf.out"


# ternary
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='full-tnn' | tee "$LOGDIR/tt.out"
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='qtnn' abits=2 | tee "$LOGDIR/t2.out"
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='qtnn' abits=4 | tee "$LOGDIR/t4.out"
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='qtnn' abits=8 | tee "$LOGDIR/t8.out"
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='tnn' abits=8 | tee "$LOGDIR/tf.out"


# qnn
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='full-qnn' wbits=4 abits=2 | tee "$LOGDIR/42.out"
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='full-qnn' wbits=4 abits=4 | tee "$LOGDIR/44.out"
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='full-qnn' wbits=4 abits=8 | tee "$LOGDIR/48.out"
python3 train.py -c "$CONFIG" -o date="$TIME" lr="$LR" nres="$NRES" cuda="$GPU" network_type='qnn' wbits=4 | tee "$LOGDIR/4f.out"
