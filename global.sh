NRES=$1
GPU=$2

./train.sh "$NRES" 0.001 "$GPU"
./train.sh "$NRES" 0.005 "$GPU"
./train.sh "$NRES" 0.01 "$GPU"
./train.sh "$NRES" 0.02 "$GPU"
./train.sh "$NRES" 0.1 "$GPU"
