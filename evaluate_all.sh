#!/bin/bash

# Evaluate all the models
PATH_TO_MODELS='./trained models/simclr_20240207_102210'

for model in $(ls "$PATH_TO_MODELS"); do
    # check for h5 files
    if [[ $model != *".h5"* ]]; then
        continue
    fi
    echo "Evaluating model $model"
    destination=$PATH_TO_MODELS/$(echo $model | cut -d'.' -f1)"_npd10balanced"
    python linear_evaluation.py \
        --pretrained_weights "$PATH_TO_MODELS/$model" \
        --path_to_imagefolder "/Users/ima029/Desktop/SCAMPI/Repository/data/labelled crops (NPD)/genera_level/imagefolder10classes_balanced" \
        --destination "$destination"
done
```

```bash