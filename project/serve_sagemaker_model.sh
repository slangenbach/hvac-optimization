MODEL_PATH="file:///Users/stefan/Code/hvac-optimization/models"
MODEL_NAME="tree_limited"
MODEL=$MODEL_PATH/$MODEL_NAME
    
mlflow sagemaker run-local \
    -m $MODEL \
    -p 5138 \
    -i $MODEL_NAME