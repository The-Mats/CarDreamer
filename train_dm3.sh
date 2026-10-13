#!/bin/bash

# Check if a port argument is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <carla_port> <gpu_device> [additional_training_parameters]"
    exit 1
fi

# Configuration
CARLA_PORT=$1
GPU_DEVICE=$2
LOG_FILE="log_${CARLA_PORT}.log"
# Keep everything on the one card named by $GPU_DEVICE (an nvidia-smi index).
# CUDA_DEVICE_ORDER makes CUDA number its devices like nvidia-smi does.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# CARLA renders through Vulkan, which CUDA_VISIBLE_DEVICES does not filter, and
# Vulkan's device order is not PCI order - so -graphicsadapter needs Vulkan's own
# index for this card. Resolve it by matching PCI bus numbers (nvidia-smi reports
# the bus in hex, vulkaninfo in decimal).
TARGET_BUS=$((16#$(nvidia-smi --query-gpu=pci.bus_id -i "$GPU_DEVICE" --format=csv,noheader | cut -d: -f2)))
VULKAN_ADAPTER=$(vulkaninfo 2>/dev/null | awk -v t="$TARGET_BUS" '
    /^GPU[0-9]+:/ { gsub(/[^0-9]/, "", $1); gpu = $1 }
    /pciBus/      { if ($3 + 0 == t) { print gpu; exit } }')
if [ -z "$VULKAN_ADAPTER" ]; then
    echo "WARNING: could not resolve Vulkan adapter for GPU $GPU_DEVICE (is vulkan-tools installed?); falling back to $GPU_DEVICE" >&2
    VULKAN_ADAPTER=$GPU_DEVICE
fi

CARLA_SERVER_COMMAND="$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -graphicsadapter=$VULKAN_ADAPTER -carla-port=$CARLA_PORT -benchmark -fps=10"
# Pattern used to detect a running CARLA; derived so it stays in sync with the command above.
CARLA_PATTERN="CarlaUE4.sh -RenderOffScreen -graphicsadapter=$VULKAN_ADAPTER -carla-port=$CARLA_PORT -benchmark -fps=10"
TRAINING_SCRIPT="dreamerv3/train.py"
# CUDA_VISIBLE_DEVICES hides the other cards from JAX, which would otherwise
# reserve a few hundred MB on each; the one visible card is then JAX's device 0.
COMMON_PARAMS="--env.world.carla_port $CARLA_PORT --dreamerv3.jax.policy_devices 0 --dreamerv3.jax.train_devices 0"
ADDITIONAL_PARAMS="${@:3}"  # Capture all additional parameters passed to the script
TRAINING_COMMAND="env CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $TRAINING_SCRIPT $COMMON_PARAMS $ADDITIONAL_PARAMS"
# Give up after this many consecutive crashes. A crash only counts as consecutive
# if training died within MIN_UPTIME seconds of starting; longer runs reset the count.
MAX_RESTARTS=${MAX_RESTARTS:-5}
MIN_UPTIME=${MIN_UPTIME:-600}
CRASH_COUNT=0

# Clear log file before starting
> $LOG_FILE

# Function to log messages with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
}

# Function to start or restart CARLA
launch_carla() {
    # Check if CARLA is running
    if ! pgrep -f "$CARLA_PATTERN" > /dev/null; then
        log_with_timestamp "CARLA server is not running on port $CARLA_PORT. Starting or restarting..."
        # Kill any existing CARLA processes on the same port
        fuser -k ${CARLA_PORT}/tcp
        # Start CARLA
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE $CARLA_SERVER_COMMAND &
        # Wait for CARLA to fully start
        while ! nc -z localhost $CARLA_PORT; do
            log_with_timestamp "Waiting for CARLA server to start on port $CARLA_PORT..."
            sleep 1  # delay to prevent excessive resource usage
        done
        log_with_timestamp "CARLA server is up and running on port $CARLA_PORT."
    fi
}

# Function to start the training script
start_training() {
    launch_carla
    # Start the training script
    $TRAINING_COMMAND >> $LOG_FILE 2>&1 &
    TRAINING_PID=$!
    TRAINING_START=$(date +%s)
    # Log the information about the log file
    log_with_timestamp "Training session started successfully. Logs are being written to: $LOG_FILE"
    echo -e "\033[1;32mTraining session started successfully. Logs are being written to: $LOG_FILE\033[0m"
}

# Function to clean up processes on exit
cleanup() {
    log_with_timestamp "Cleaning up and exiting..."
    # Kill CARLA process
    fuser -k ${CARLA_PORT}/tcp
    # Kill the specific training process using its PID
    kill -TERM $TRAINING_PID >/dev/null 2>&1
    wait $TRAINING_PID >/dev/null 2>&1
    exit
}

# Trap EXIT signal to call the cleanup function
trap cleanup SIGINT

# Initial start
log_with_timestamp "Starting training on port $CARLA_PORT..."
log_with_timestamp "Training command: $TRAINING_COMMAND"
start_training

while true; do
    # Check if the training script is still running
    if ! pgrep -f "$TRAINING_SCRIPT" > /dev/null; then
        if (( $(date +%s) - TRAINING_START < MIN_UPTIME )); then
            CRASH_COUNT=$((CRASH_COUNT + 1))
        else
            CRASH_COUNT=1
        fi
        if (( CRASH_COUNT >= MAX_RESTARTS )); then
            log_with_timestamp "Training script crashed $CRASH_COUNT times in a row on port $CARLA_PORT. Giving up."
            echo -e "\033[1;31mTraining crashed $CRASH_COUNT times in a row. Giving up - check $LOG_FILE for the error.\033[0m"
            cleanup
        fi
        log_with_timestamp "Training script crashed on port $CARLA_PORT ($CRASH_COUNT/$MAX_RESTARTS). Restarting..."
        start_training
    fi
    # Check if CARLA server needs to be restarted
    launch_carla
    # Check every minute
    sleep 60
done
