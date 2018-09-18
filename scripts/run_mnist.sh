set -x

EXP_NAME=$1
TOP_K=$2
POOL_SIZE=$3
LR=$4
DECAY=0

OUTPUT_DIR="/proj/BigLearning/ahjiang/output/mnist/"$EXP_NAME
OUTPUT_FILE="mnist_lecunn_"$TOP_K"_"$POOL_SIZE"_"$LR"_"$DECAY"_v1"
PICKLE_DIR=$OUTPUT_DIR/pickles

mkdir $OUTPUT_DIR
mkdir $PICKLE_DIR

NUM_TRIALS=1
for i in `seq 1 $NUM_TRIALS`
do
  OUTPUT_FILE="mnist_lecunn_"$TOP_K"_"$POOL_SIZE"_"$LR"_"$DECAY"_trial"$i"_v1"
  PICKLE_PREFIX="mnist_lecunn_"$TOP_K"_"$POOL_SIZE"_"$LR"_"$DECAY"_trial"$i

  python mnist/main.py \
    --batch-size 1 \
    --selective-backprop=True \
    --top-k $TOP_K \
    --pool-size $POOL_SIZE \
    --decay $DECAY \
    --pickle-dir=$PICKLE_DIR \
    --pickle-prefix=$PICKLE_PREFIX \
    --lr $LR &> $OUTPUT_DIR/$OUTPUT_FILE
done
