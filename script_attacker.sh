# Default values
NB_WORKERS=""
BATCH_SIZE=""
NB_AGENTS=""
NB_MACHINES=""
EXPERIMENT_DIR=""

# Usage function to show how to use the script
usage() {
    echo "Usage: $0 --experiment_dir <EXPERIMENT_DIR> --nb_workers <NB_WORKERS> --batch_size <BATCH_SIZE>"
    exit 1
}

# Parse the arguments using a while loop
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment_dir) EXPERIMENT_DIR="$2"; shift ;;
        --nb_workers) NB_WORKERS="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Check if all required parameters are set
if [[ -z "$EXPERIMENT_DIR" || -z "$NB_WORKERS" || -z "$BATCH_SIZE" ]]; then
    echo "Error: Missing required parameters."
    usage
fi


# Check if DECENTRALIZEPY_G5K_DIR is set
if [ -z "${DECENTRALIZEPY_G5K_DIR}" ]; then
    echo "DECENTRALIZEPY_G5K_DIR is not set"
    # Default value for Jean-Zay.
    export DECENTRALIZEPY_G5K_DIR=$WORK/decentralizepy_grid5000
fi
echo "DECENTRALIZEPY_G5K_DIR: ${DECENTRALIZEPY_G5K_DIR}"

DECENTRALIZEPY_DIR="${DECENTRALIZEPY_G5K_DIR}/decentralizepy"
echo "DECENTRALIZEPY_DIR: ${DECENTRALIZEPY_DIR}"

# Check if DATASETS_DIR is set
if [ -z "${DATASETS_DIR}" ]; then
    echo "DATASETS_DIR is not set"
    # Default value for Jean-Zay.
    export DATASETS_DIR=$SCRATCH/datasets
fi
echo "DATASETS_DIR: ${DATASETS_DIR}"

# Check if CONTAINER_FILE is set
if [ -z "${CONTAINER_FILE}" ]; then
    echo "DATASETS_DIR is not set"
    # Default value for Jean-Zay.
    export CONTAINER_FILE=$SINGULARITY_ALLOWED_DIR/attacker_container.sif
fi
echo "CONTAINER_FILE: ${CONTAINER_FILE}"



# Define the base command
BASE_COMMAND="singularity run \
    --bind '$EXPERIMENT_DIR:/experiments_to_attack' \
    --bind $DATASETS_DIR:/datasets \
    --bind $DECENTRALIZEPY_G5K_DIR:/decentralizepy_grid5000 \
    --bind $DECENTRALIZEPY_DIR:/decentralizepy \
    --nv \
    $CONTAINER_FILE \
    --nb_workers $NB_WORKERS \
    --batch_size $BATCH_SIZE \
    --datasets_dir /datasets"  # The datasets folder is binded to /datasets.

# Conditionally add 'srun' if using SLURM
if [ "${USE_SLURM}" == "true" ]; then
    COMMAND="srun $BASE_COMMAND"
else
    COMMAND="$BASE_COMMAND"
fi


# Build and run the Apptainer (or Singularity) container command
eval $COMMAND
