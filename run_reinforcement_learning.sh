NUM_ACTORS=$1
GPU_USE=$2
PRE_TRAINED_MODEL=$3

tmux new-session -d -t impala_minecraft

tmux new-window -d -n learner
COMMAND_LEARNER='python3.7 learner.py --env_num '"${NUM_ACTORS}"' --gpu_use '"${GPU_USE}"' --pretrained_model '"${PRE_TRAINED_MODEL}"''
#COMMAND_LEARNER='python3.7 learner.py --env_num '"${NUM_ACTORS}"' --gpu_use '"${GPU_USE}"''
echo $COMMAND_LEARNER

tmux send-keys -t "learner" "$COMMAND_LEARNER" ENTER

sleep 10

for ((id=0; id < $NUM_ACTORS; id++)); do
    tmux new-window -d -n "actor_${id}"
    COMMAND='MINERL_HEADLESS=1 python3.7 actor.py --env_id  '"${id}"''
    tmux send-keys -t "actor_${id}" "$COMMAND" ENTER

    sleep 0.5
done

tmux attach -t impala_minecraft