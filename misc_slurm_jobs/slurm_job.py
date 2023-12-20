"""Contains basic helper functions for running a parameter sweep on the Hyak
cluster using the SLURM scheduler.
Adapted from ParlAI
"""

from collections import namedtuple
import json
import os
import subprocess
import sys
import random
import hashlib

from slurm_constants import CONSTANTS
username = ""
RUN_CONSTANTS = CONSTANTS.get(username)
if RUN_CONSTANTS is None:
    raise Error("username isn't defined in slurm_constants file")

DEFAULT_DIR_PATH = '/gscratch/zlab/margsli/gitfiles/demix'
BASH_IF_CLAUSE = """
if [[ "$SLURM_ARRAY_TASK_ID" == "{index}" ]]; then
    srun -K1 bash {SAVE}/run.sh > {SAVE}/stdout.$SLURM_ARRAY_TASK_ID 2> {SAVE}/stderr.$SLURM_ARRAY_TASK_ID
fi
"""
SLRM_JOB_ARRAY_TEMPLATE = """
#!/bin/bash
#SBATCH --job-name={SWEEP_NAME}
#SBATCH --output={SAVE_ROOT}/slurm_logs/stdout.%j
#SBATCH --error={SAVE_ROOT}/slurm_logs/stderr.%j
#SBATCH --account={account}
#SBATCH --partition={partition}
## make sure we don't clobber log files if jobs get restarted
#SBATCH --open-mode=append
#SBATCH --nodes={nodes}
#SBATCH --time={jobtime}
## make sure we are told about preempts, and jobs running out of time, 5 min beforehand
#SBATCH --signal=USR1@60
## number of cpus *per task*. Highly recommend this to be 10.
#SBATCH --cpus-per-task={cpus}
## srun forks ntasks_per_node times on each node
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --mem={mem_gb}G
{SBATCH_EXTRAS}

source ~/.bashrc
{conda_command}

echo "# -------- BEGIN CALL TO run.sh --------"
# -K kills all subtasks if one particular task crashes. This is necessary for
# distributed training
{JOB_LAUNCHER}
"""

SH_TEMPLATE = """
#!/bin/bash
set -e

# stores the child process
CHILD=""

# handles a TERM signal
term_handler () {{
    # catch and ignore TERM. we get multiple terms during shutdown, so best
    # to just do nothing
    # but still keep going with the python process
    wait "$CHILD"
}}

# handles an interrupt (aka ctrl-C)
int_handler () {{
    # propagate a ctrl-C to the python process
    kill -s INT "$CHILD"
    wait "$CHILD"
}}

# handles a USR1, which signals preemption or job time is up
usr1_handler () {{
    echo "SLURM signaling preemption/times up (SLURM_PROCID $SLURM_PROCID)."
    kill -s INT "$CHILD"  # send ctrl-c to python
    if {SHOULD_REQUEUE} && [ "$SLURM_PROCID" -eq "0" ]; then
        echo "Waiting 5s and resubmitting..."
        sleep 5
        echo "Resubmitting..."
        scontrol requeue $SLURM_JOB_ID
    fi
    wait "$CHILD"
}}

trap 'int_handler' INT
trap 'usr1_handler' USR1
trap 'term_handler' TERM

# Uncommenting these two lines can help with identifying hangs
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# setting this this can also help with hangs
# NCCL_LL_THRESHOLD=0

# if in distributed, make sure we're using the actual network
export NCCL_SOCKET_IFNAME=^docker0,lo
echo
nvidia-smi

cd {NEW_DIR_PATH}
export PYTHONPATH={SAVE_ROOT}/demix:$PYTHONPATH
{python_cmd} &
echo "# -------- FINISHED CALL TO SRUN --------"
echo
nvidia-smi

CHILD="$!"
wait "$CHILD"
sleep 30
"""


def sha1(string):
    """Compute the sha1 hexdigest of the string."""
    return hashlib.sha1(string.encode('utf-8')).hexdigest()


def run_grid(
    grid,
    name_keys,
    sweep_name,
    user=os.environ['USER'],
    prefix=None,
    gpus=1,
    cpus=10,
    nodes=1,
    node_exclude=None,
    account='',
    partition='gpu-rtx6k',
    DIR_PATH=DEFAULT_DIR_PATH,
    jobtime='01:59:59',
    include_job_id=False,
    hide_keys={},
    hashname=False,
    fixedname=False,
    saveroot=None,
    logroot=None,
    mem_gb=64,
    requeue=False,
    data_parallel=False,
    comment=None,
    volta=False,
    volta32=False,
    copy_env=True,
    copy_dirs=[],
    max_num_jobs=None,
    num_copies=1,
    job_id_start=1,
    debug_mode=False,
    dry_mode=False,
    add_name=None,
    dependencies=[],
    constraints=[],
    conda_env_name="time_weights_env",
):
    """Generates full commands from a grid.

    Arguments:
    grid -- (dict) keys are hyperparam strings (e.g. --learningrate or -lr),
        values are lists of parameter options (e.g. [0.5, 0.05, 0.005]).
        You can tie options together in a limited fashion (e.g.
        '--opt': ['sgd -lr 0.5', 'adam -lr 0.005']), but we don't support
        nesting dicts/lists yet.
    name_keys -- (set) contains any params to always include in the model
        filename (e.g. {'-hs'} will make sure that the filename includes
        _hs=X_). By default, any key with more than one value will also be
        included in the model filename.
    sweep_name -- (str) name of the sweep
    user -- (str) user name to use for save directory (default $USER)
    prefix -- (str) base command to run
    hashname -- (bool) if True, uses a hash of the parameters as the
        folder. Sometimes necessary for long commands (default False).
    dataparallel -- (bool) set to True if running with nn.DataParallel
    volta -- (bool) set to True to request a volta machine
    volta32 -- (bool) set to True to request a 32gb volta machine
    comment -- you need to add a text comment to use priority partition
    copy_env -- (bool) if True, copies local directory components to the
        save root, and uses this to run the jobs
    copy_dirs -- (list) list of additional directories to copy
    max_num_jobs -- (int) maximum number of jobs
    add_name -- (str) "end" or None, indicating whether to
        add the name to the command and if so, where
    """
    if not prefix:
        raise ValueError('Need prefix command')
    # if not hasattr(grid, 'items'):
    #     raise TypeError('Grid should be a dict.')

    if saveroot is None:
        SAVE_ROOT = save_root(sweep_name, user)
    else:
        SAVE_ROOT = saveroot
    if logroot is None:
        LOG_ROOT = log_root(sweep_name, user)
    else:
        LOG_ROOT = logroot
    Job = namedtuple('Job', ['cmd', 'name'])
    all_jobs = [Job(cmd=prefix, name='')]

    for key, args in grid.get('positional_args', {}).items():
        new_jobs = []
        # save_name
        save_key = key
        while save_key.startswith('-'):
            save_key = save_key[1:]
        save_key = save_key.replace('_', '')

        for job in all_jobs:
            for a in args:
                new_cmd = ' '.join((job.cmd, str(a)))
                new_name = job.name
                if (len(args) > 1 or key in name_keys) and key not in hide_keys:
                    if a is None:
                        new_jobs.append(Job(cmd=new_cmd, name=new_name))
                        continue
                    if type(a) == str:
                        a = a.replace('_', '')
                        if ' ' in a:
                            a = a.replace(' --', '_').replace(' -', '_')
                            a = a.replace(' ', '=')
                    new_name += '_{}={}'.format(save_key, a)
                if hashname:
                    new_name = sha1(new_name)
                new_jobs.append(Job(cmd=new_cmd, name=new_name))
        all_jobs = new_jobs

    if grid.get('fixed_args'):
        new_jobs = []
        for job in all_jobs:
            new_cmd = ' '.join((job.cmd, fixed_args))
            new_name = job.name
            new_jobs.append(Job(cmd=new_cmd, name=new_name))
        all_jobs = new_jobs

    for key, args in grid.get('named_args', {}).items():
        new_jobs = []
        # save_name
        save_key = key
        while save_key.startswith('-'):
            save_key = save_key[1:]
        save_key = save_key.replace('_', '')

        for job in all_jobs:
            for a in args:
                new_cmd = ' '.join((job.cmd, str(key), str(a)))
                new_name = job.name
                if (len(args) > 1 or key in name_keys) and key not in hide_keys:
                    if len(a) == 0:
                        new_jobs.append(Job(cmd=new_cmd, name=new_name))
                        continue
                    if type(a) == str:
                        a = a.replace('_', '')
                        if ' ' in a:
                            a = a.replace(' --', '_').replace(' -', '_')
                            a = a.replace(' ', '=')
                    new_name += '_{}={}'.format(save_key, a)
                if hashname:
                    new_name = sha1(new_name)
                new_jobs.append(Job(cmd=new_cmd, name=new_name))
        all_jobs = new_jobs

    # if add_name:
    #     new_jobs = []
    #     for job in all_jobs:
    #         new_cmd = ' '.join((job.cmd, job.name))
    #         new_name = job.name
    #         new_jobs.append(Job(cmd=new_cmd, name=new_name))
    #     all_jobs = new_jobs

    # Sample grid points
    if isinstance(max_num_jobs, int) and max_num_jobs < len(all_jobs):
        all_jobs = random.sample(all_jobs, max_num_jobs)

    # shorten names if possible
    if hashname:
        # keep the names from getting too long
        full_names = [name for _, name in all_jobs]
        cutoff = i = 4
        while i < 40:
            if len(set([n[1:i] for n in full_names])) == len(full_names):
                cutoff = i
                break
            i += 1
    else:
        cutoff = None

    final_jobs = []
    job_id = job_id_start
    for job in all_jobs:
        for _ in range(num_copies):
            new_cmd = job.cmd
            new_name = job.name[1:cutoff] if cutoff else job.name[1:]
            if include_job_id:
                if fixedname:
                    new_name = fixedname
                new_name += '/_jobid=' + str(job_id)
            # else:
            #     new_cmd = '{} '.format(job.cmd)
            if add_name:
                new_cmd = ' '.join((new_cmd, new_name))
            final_jobs.append(Job(cmd=new_cmd, name=new_name))
            job_id += 1

    print('Example of first job:\n{}\n'.format(final_jobs[0].cmd))
    if dry_mode:
        return

    print('Your jobs will run for {}.'.format(jobtime))
    # ans = input(
    #     'About to launch {} jobs for a total of {} GPUs. Continue? (Y/y to proceed) '.format(
    #         len(final_jobs), nodes * gpus * len(final_jobs)
    #     )
    # )
    # if ans.strip().lower() != 'y':
    #     print('Aborting...')
    #     sys.exit(-1)

    if copy_env:
        bash('mkdir -p ' + os.path.join(SAVE_ROOT, 'demix'))
        to_copy = []
        to_copy += copy_dirs
        for c in to_copy:
            c_head, c_tail = os.path.split(c)
            # if subfolder, copy folder then subfolder
            if len(c_head) > 1:
                bash('mkdir {SAVE_ROOT}/demix/{c_head}'.format(**locals()))
            bash('cp -r {DIR_PATH}/{c} {SAVE_ROOT}/demix/{c}'.format(**locals()))
        NEW_DIR_PATH = '{SAVE_ROOT}/demix'.format(**locals())
    else:
        NEW_DIR_PATH = DIR_PATH

    # Dump grid to grid file
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)
    with open(os.path.join(SAVE_ROOT, 'grid.json'), 'w') as f:
        json.dump(grid, f)

    # shuffle jobs so we're not systematically doing them in any order
    # random.shuffle(final_jobs)
    # remove job array list if it already existed
    jobs_path = []
    if debug_mode and len(final_jobs) > 1:
        final_jobs = final_jobs[:1]
    for job in final_jobs:
        jobs_path.append(
            create_job_files(
                sweep_name,
                SAVE_ROOT,
                LOG_ROOT,
                job.name,
                job.cmd,
                gpus=gpus,
                nodes=nodes,
                data_parallel=data_parallel,
                requeue=requeue,
                NEW_DIR_PATH=NEW_DIR_PATH,
            )
        )
    print(final_jobs)
    submit_array_jobs(
        SWEEP_NAME=sweep_name,
        SAVE_ROOT=SAVE_ROOT,
        gpus=gpus,
        cpus=cpus,
        nodes=nodes,
        node_exclude=node_exclude,
        account=account,
        partition=partition,
        jobtime=jobtime,
        DIR_PATH=DIR_PATH,
        mem_gb=mem_gb,
        requeue=requeue,
        data_parallel=data_parallel,
        comment=comment,
        volta=volta,
        volta32=volta32,
        NEW_DIR_PATH=NEW_DIR_PATH,
        jobs_path=jobs_path,
        dependencies=dependencies,
        constraints=constraints,
        conda_env_name=conda_env_name,
    )


def bash(bashCommand):
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = str(output)
    output = output[:-3]
    output = output.lstrip('b').strip('\'').strip('"')
    return output


def save_root(SWEEP_NAME, unixname):
    """Return root folder for saving model files, stdout, stderr, etc."""
    # DATE = bash('date +"%Y%m%d"')
    # SAVE_ROOT = os.path.join(RUN_CONSTANTS.get('MODEL_FOLDER'), DATE, SWEEP_NAME)
    SAVE_ROOT = os.path.join(RUN_CONSTANTS.get('MODEL_FOLDER'), SWEEP_NAME)
    return SAVE_ROOT

def log_root(SWEEP_NAME, unixname):
    """Return root folder for saving tensorboard logs"""
    # DATE = bash('date +"%Y%m%d"')
    # SAVE_ROOT = os.path.join('/gscratch/zlab/margsli/demix-checkpoints', DATE, SWEEP_NAME)
    LOG_ROOT = os.path.join(RUN_CONSTANTS.get('LOG_FOLDER'), SWEEP_NAME)
    return LOG_ROOT


def create_job_files(
    SWEEP_NAME,
    SAVE_ROOT,
    LOG_ROOT,
    job_name,
    python_cmd,
    gpus=1,
    nodes=1,
    data_parallel=False,
    requeue=False,
    NEW_DIR_PATH=DEFAULT_DIR_PATH,
):
    """Creates job folders and scripts"""
    SHOULD_REQUEUE = str(requeue).lower()
    SAVE = os.path.join(SAVE_ROOT, job_name)[:250]
    bash('mkdir -p ' + SAVE)
    LOG = os.path.join(LOG_ROOT, job_name)
    bash('mkdir -p ' + LOG)
    SCRIPTFILE = os.path.join(SAVE, 'run.sh')
    ARRAYJOBFILE = os.path.join(SAVE_ROOT, 'array_jobs')

    if data_parallel or not gpus:
        ntasks_per_node = 1
    else:
        ntasks_per_node = gpus
    with open(SCRIPTFILE, 'w') as fw:
        fw.write(SH_TEMPLATE.format(**locals()).lstrip())
    return SAVE


def submit_array_jobs(
    SWEEP_NAME,
    SAVE_ROOT,
    gpus=1,
    cpus=1,
    nodes=1,
    node_exclude=None,
    account='zlab',
    partition='gpu-rtx6k',
    jobtime='23:59:59',
    DIR_PATH=DEFAULT_DIR_PATH,
    mem_gb=64,
    requeue=False,
    data_parallel=False,
    comment=None,
    volta=False,
    volta32=False,
    NEW_DIR_PATH=DEFAULT_DIR_PATH,
    jobs_path=[],
    dependencies=[],
    constraints = [],
    conda_env_name="time_weights_env",
):
    SLURMFILE = os.path.join(SAVE_ROOT, 'run.slrm')
    if data_parallel or not gpus:
        ntasks_per_node = 1
    else:
        ntasks_per_node = gpus
    SBATCH_EXTRAS = []
    if node_exclude is not None:
        # If any nodes are down, exclude them here
        SBATCH_EXTRAS.append('#SBATCH --exclude ' + str(node_exclude))

    if volta32:
        constraints.append('volta32gb')

    total_num_jobs = len(jobs_path) - 1

    # Request the number of GPUs (defaults to 1)
    gpustr = '#SBATCH --gpus-per-node={}'.format(gpus)
    SBATCH_EXTRAS.append(gpustr)

    if constraints:
        SBATCH_EXTRAS.append("#SBATCH -C '{}'".format('&'.join(constraints)))
    
    
    if comment:
        SBATCH_EXTRAS.append('#SBATCH --comment="{}"'.format(comment))

    if dependencies:
        SBATCH_EXTRAS.append('#SBATCH --dependency="{}"'.format(','.join(['afterok:' + str(d) for d in dependencies])))

    conda_command = f'conda activate {conda_env_name}' if conda_env_name else ''

    # make sure sbatch extras are a string
    SBATCH_EXTRAS = "\n".join(SBATCH_EXTRAS)
    JOB_LAUNCHER = []
    for idx, each_path in enumerate(jobs_path):
        JOB_LAUNCHER.append(BASH_IF_CLAUSE.format(index=idx, SAVE=each_path))
    JOB_LAUNCHER = "\n".join(JOB_LAUNCHER)
    bash('mkdir -p ' + os.path.join(SAVE_ROOT, 'slurm_logs'))
    with open(SLURMFILE, 'w') as fw:
        fw.write(SLRM_JOB_ARRAY_TEMPLATE.format(**locals()).lstrip())
        
    print(bash('sbatch --array=0-{} {}'.format(total_num_jobs, SLURMFILE)))
