MODEL_PATH="file:///Users/stefan/Code/hvac-optimization/models"
MODEL_NAME="tree_limited"
MODEL=$MODEL_PATH/$MODEL_NAME/

mlflow models serve \
    -m $MODEL \
    -p 5137 \
    --no-conda