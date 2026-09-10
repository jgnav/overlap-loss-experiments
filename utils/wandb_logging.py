"""Shared YAML-controlled W&B initialization for training and evaluation."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WANDB_DEFAULTS = {
    "wandb_mode": "disabled",
    "wandb_project": None,
    "wandb_entity": None,
    "wandb_run_name": None,
    "wandb_run_id": None,
    "wandb_resume": None,
}


def configure_wandb(values):
    settings = {**WANDB_DEFAULTS, **{k: values[k] for k in WANDB_DEFAULTS if k in values}}
    if settings["wandb_mode"] not in ("online", "offline", "disabled"):
        raise ValueError("wandb_mode must be online, offline, or disabled")
    for name in ("wandb_project", "wandb_entity", "wandb_run_name", "wandb_run_id"):
        value = settings[name]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a nonempty string or null")
    if settings["wandb_mode"] != "disabled" and settings["wandb_project"] is None:
        raise ValueError("wandb_project is required when W&B is enabled")
    if settings["wandb_resume"] not in (None, "allow", "must", "never"):
        raise ValueError("wandb_resume must be null, allow, must, or never")
    if settings["wandb_resume"] and not settings["wandb_run_id"]:
        raise ValueError("wandb_resume requires wandb_run_id in the YAML")
    if settings["wandb_run_id"] and settings["wandb_resume"] is None:
        settings["wandb_resume"] = "must"
    values.update(settings)
    return values


def init_wandb_run(args, config, job_type):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    settings = None
    if args.wandb_mode == "online":
        key_path = REPO_ROOT / ".wandb_key"
        try:
            key = key_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"W&B key file not found: {key_path}") from None
        if not key or any(character.isspace() for character in key):
            raise ValueError(".wandb_key must contain one nonempty W&B API key")
        # Pass authentication directly to the SDK, never into args/config,
        # checkpoints, command-line arguments, or the workers' environment.
        settings = wandb.Settings(api_key=key)
    return wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name,
        mode=args.wandb_mode, id=args.wandb_run_id, resume=args.wandb_resume,
        dir=str(args.output_dir), config=config, job_type=job_type, settings=settings,
    )


def log_evaluation(run, name, result):
    if run is None:
        return
    metrics = {f"{name}/{key}": value for key, value in result.get("metrics", {}).items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)}
    if metrics:
        run.log(metrics)
        run.summary.update(metrics)
