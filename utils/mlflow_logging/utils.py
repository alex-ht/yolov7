import os
import mlflow

def log_parameters(hyp, opt):
    for k, v in hyp.items():
        mlflow.log_param(k, v)
    for key, value in vars(opt).items():
        mlflow.log_param(key, value)

def log_model(model, opt, epoch, fitness_score, best_model=False):
    mlflow.pytorch.log_model(model, artifact_path="last.pt")
    if best_model:
        mlflow.pytorch.log_model(model, artifact_path="best.pt")
    mlflow.pytorch.log_model(model, artifact_path=f"epoch{epoch+1}.pt")
    print("The model is logged at:\n%s" % (os.path.join(mlflow.get_artifact_uri(), "last.pt")))
    print("Saving model artifact on epoch ", epoch + 1)