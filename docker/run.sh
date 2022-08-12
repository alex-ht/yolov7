#!/bin/bash
set -e
[ -d data ] && rm -rf data && echo "clean data/"
tar xf $INPUT/data.tar
trap "rm -rf data" EXIT

find $(pwd)/data/train -iname "*g" | sort > $OUTPUT/train.txt
find $(pwd)/data/valid -iname "*g" | sort > $OUTPUT/valid.txt

echo "train: $OUTPUT/train.txt" >  $OUTPUT/small_lp.yaml
echo "val: $OUTPUT/valid.txt"   >> $OUTPUT/small_lp.yaml
echo "test: $OUTPUT/valid.txt"  >> $OUTPUT/small_lp.yaml
echo "nc: 1"                    >> $OUTPUT/small_lp.yaml
echo "names: [ 'lp' ]"          >> $OUTPUT/small_lp.yaml

python /yolov7/train.py \
  --workers $WORKERS \
  --device 0 \
  --batch-size $BATCH_SIZE \
  --data $OUTPUT/small_lp.yaml \
  --img $IMG $IMG \
  --cfg $INPUT/$CFG \
  --weights "$INPUT/$WEIGHTS" \
  --name $NAME \
  --hyp $INPUT/$HYP \
  --project $OUTPUT \
  --epochs ${EPOCHS} \
  --cache-images

exit 0