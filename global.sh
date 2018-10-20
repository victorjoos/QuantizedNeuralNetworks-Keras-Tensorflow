NRES=$1
GPU=$2
PFI=$3

# ./train.sh "$NRES" 0.001 "$GPU" "$PFI"
# ./train.sh "$NRES" 0.005 "$GPU" "$PFI"
./train.sh "$NRES" 0.01 "$GPU" "$PFI"
./train.sh "$NRES" 0.02 "$GPU" "$PFI"
# ./train.sh "$NRES" 0.1 "$GPU" "$PFI"
