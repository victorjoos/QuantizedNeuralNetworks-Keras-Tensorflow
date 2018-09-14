#!/usr/bin/env bash
# cuDNN

CONFIG=config
TIME=$(date +"%Y-%m-%d_%H-%M-%S")
mkdir "logs/$TIME"
# standard float
python3 train.py -c "$CONFIG" -o network_type='float' | tee "logs/$TIME/ff.out"

# binary
python3 train.py -c "$CONFIG" -o network_type='full-bnn' | tee "logs/$TIME/bb.out"
python3 train.py -c "$CONFIG" -o network_type='qbnn' abits=2 | tee "logs/$TIME/b2.out"
python3 train.py -c "$CONFIG" -o network_type='qbnn' abits=4 | tee "logs/$TIME/b4.out"
python3 train.py -c "$CONFIG" -o network_type='qbnn' abits=8 | tee "logs/$TIME/b8.out"
python3 train.py -c "$CONFIG" -o network_type='bnn' abits=8 | tee "logs/$TIME/bf.out"


# ternary
python3 train.py -c "$CONFIG" -o network_type='full-tnn' | tee "logs/$TIME/tt.out"
python3 train.py -c "$CONFIG" -o network_type='qtnn' abits=2 | tee "logs/$TIME/t2.out"
python3 train.py -c "$CONFIG" -o network_type='qtnn' abits=4 | tee "logs/$TIME/t4.out"
python3 train.py -c "$CONFIG" -o network_type='qtnn' abits=8 | tee "logs/$TIME/t8.out"
python3 train.py -c "$CONFIG" -o network_type='tnn' abits=8 | tee "logs/$TIME/tf.out"


# qnn
python3 train.py -c "$CONFIG" -o network_type='full-qnn' wbits=4 abits=2 | tee "logs/$TIME/42.out"
python3 train.py -c "$CONFIG" -o network_type='full-qnn' wbits=4 abits=4 | tee "logs/$TIME/44.out"
python3 train.py -c "$CONFIG" -o network_type='full-qnn' wbits=4 abits=8 | tee "logs/$TIME/48.out"
python3 train.py -c "$CONFIG" -o network_type='qnn' wbits=4 | tee "$TIME/4f.out"
