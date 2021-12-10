NUM_ACTORS=$1
WORKSPACE_PATH=$2

tmux new-session -d -t impala_pong

tmux new-window -d -n learner
COMMAND='python3.7 learner.py --env_number '"${NUM_ACTORS}"' --workspace_path '"${WORKSPACE_PATH}"''
echo $COMMAND
tmux send-keys -t "learner" "$COMMAND" ENTER

for ((id=0; id < $NUM_ACTORS; id++)); do
    tmux new-window -d -n "actor_${id}"
    COMMAND='python3.7 actor.py --env_id  '"${id}"''
    tmux send-keys -t "actor_${id}" "$COMMAND" ENTER
done

tmux attach -t impala_pong