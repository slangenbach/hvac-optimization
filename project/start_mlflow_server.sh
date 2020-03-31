BACKEND_URI="file:///Users/stefan/Code/hvac-optimization/experiments"
ARTIFACT_ROOT="file:///Users/stefan/Code/hvac-optimization/references"

mlflow server \
    --backend-store-uri $BACKEND_URI \
    --default-artifact-root $ARTIFACT_ROOT \
    --host localhost
    --port 5000
    
