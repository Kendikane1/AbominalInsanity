import os
import wandb


def init_run(experiment_name, config_dict, tags=None):
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gzsl-eeg-bravl"),
        entity=os.environ.get("WANDB_ENTITY", None),
        name=experiment_name,
        config=config_dict,
        tags=tags or [],
        reinit=True,
    )


def log_training_step(step, L_w, L_var=None, L_G=None, L_D=None,
                      VarR=None, norm_w=None, norm_v=None, cos_sim=None):
    payload = {"train/step": step, "train/L_wasserstein": L_w, "train/L_D": L_D}
    if L_var is not None:
        payload["train/L_var"] = L_var
    if L_G is not None:
        payload["train/L_G"] = L_G
    if VarR is not None:
        payload["train/VarR"] = VarR
    if norm_w is not None:
        payload["train/grad_norm_w"] = norm_w
        payload["train/grad_norm_v"] = norm_v
        payload["train/grad_cos_sim"] = cos_sim
    wandb.log(payload, step=step)


def log_gzsl_results(H, AccS, AccU, routing, VarR=None, rho_sp=None,
                     rho_sr=None, knn10=None, **extra):
    payload = {
        "eval/H_mean": H,
        "eval/AccS": AccS,
        "eval/AccU": AccU,
        "eval/routing_rate": routing,
    }
    if VarR is not None:
        payload["eval/VarR"] = VarR
    if rho_sp is not None:
        payload["eval/rho_sp"] = rho_sp
    if rho_sr is not None:
        payload["eval/rho_sr"] = rho_sr
    if knn10 is not None:
        payload["eval/kNN10"] = knn10
    payload.update({f"eval/{k}": v for k, v in extra.items()})
    wandb.log(payload)


def log_artifact(local_path, artifact_name, artifact_type="model"):
    art = wandb.Artifact(artifact_name, type=artifact_type)
    art.add_file(local_path)
    wandb.log_artifact(art)


def finish_run(exit_code=0):
    wandb.finish(exit_code=exit_code)
