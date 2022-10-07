
import submitit


def get_executor_jzay(
    job_name,
    timeout_hour=60,
    n_gpus=1,
    mem=3_000,
    cpus_per_task=None
):
    if timeout_hour > 20:
        qos = 't4'
    elif timeout_hour > 2:
        qos = 't3'
    else:
        qos = 'dev'

    if cpus_per_task is None:
        cpus_per_task = n_gpus * 10

    executor = submitit.AutoExecutor(job_name)
    executor.update_parameters(
        timeout_min=180,
        slurm_job_name=job_name,
        slurm_time=f'{timeout_hour}:00:00',
        slurm_gres=f'gpu:{n_gpus}',
        slurm_additional_parameters={
            'ntasks': 1,
            'cpus-per-task': cpus_per_task,
            'qos': f'qos_gpu-{qos}',
            'distribution': 'block:block',
            'mem': mem,
            # 'account': 'uwk@gpu',  # temporary - Thomas account
            'account': 'dsh@gpu',
        },
        slurm_setup=[
            '#SBATCH -C v100-32g',
            'module purge',
            'module load pytorch-gpu/py3/1.8.0',
            # 'module load pytorch-gpu/py3/1.7.1'
            'module load cuda/10.2 ',
            # 'cudnn/7.6.5.32-cuda-10.1 nccl/2.5.6-2-cuda'
        ]
    )
    return executor


def get_executor_marg(job_name, timeout_hour=60, n_cpus=10):

    executor = submitit.AutoExecutor(job_name)
    executor.update_parameters(
        timeout_min=180,
        slurm_job_name=job_name,
        slurm_time=f'{timeout_hour}:00:00',
        slurm_additional_parameters={
            'ntasks': 1,
            'cpus-per-task': n_cpus,
            'distribution': 'block:block',
        },
    )
    return executor


CLUSTER_CONFIGS = {
    'jean-zay': get_executor_jzay,
    'margaret': get_executor_marg,
}
