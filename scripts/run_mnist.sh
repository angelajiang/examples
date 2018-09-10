set -x

EXP_NAME=$1
TOP_K=$2
BATCH_SIZE=$3
LR=$4

OUTPUT_DIR="/proj/BigLearning/ahjiang/output/mnist/"$EXP_NAME
mkdir $OUTPUT_DIR
OUTPUT_FILE="mnist_cifar10_"$TOP_K"_"$BATCH_SIZE"_"$LR".v0"

python mnist/main.py \
  --batch-size $BATCH_SIZE \
  --selective-backprop=True \
  --top-k $TOP_K \
  --lr $LR &> $OUTPUT_DIR/$OUTPUT_FILE
