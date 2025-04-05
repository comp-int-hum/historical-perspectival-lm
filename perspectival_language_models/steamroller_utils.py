

def cpu_task_config(env, name, time_required, memory_required=None):
    if memory_required is None:
        memory_required = env["MEMORY"]
    return {
        "STEAMROLLER_ACCOUNT": env["CPU_ACCOUNT"],
        "STEAMROLLER_QUEUE": env["CPU_QUEUE"],
        "STEAMROLLER_TIME": time_required,
        "STEAMROLLER_MEMORY": memory_required,
        "STEAMROLLER_NAME_PREFIX": f"{name}",
        "STEAMROLLER_ENGINE": env["STEAMROLLER_ENGINE"],
    }

def gpu_task_config(env, name, time_required, memory_required=None):
    if memory_required is None:
        memory_required = env["MEMORY"]
    return {
        "STEAMROLLER_ACCOUNT": env["GPU_ACCOUNT"],
        "STEAMROLLER_QUEUE": env["GPU_QUEUE"],
        "STEAMROLLER_TIME": time_required,
        "STEAMROLLER_MEMORY": memory_required,
        "STEAMROLLER_NAME_PREFIX": f"{name}",
        "STEAMROLLER_ENGINE": env["STEAMROLLER_ENGINE"],
        "STEAMROLLER_GPU_COUNT": env["GPU_COUNT"],
    }