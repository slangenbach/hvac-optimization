MODEL_PATH="file:///Users/stefan/Code/hvac-optimization/models"
MODEL_NAME="tree_limited"
MODEL=$MODEL_PATH/$MODEL_NAME/

mlflow sagemaker build-and-push-container \
    --build \
    --no-push \
    -c $MODEL_NAME
