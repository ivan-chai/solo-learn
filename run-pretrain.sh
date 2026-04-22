DATASET=$1
METHOD=$2
shift 2

python main_pretrain.py \
       --config-path scripts/pretrain/"$DATASET"/ \
       --config-name "$METHOD".yaml "$@"
