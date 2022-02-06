NUM_ACTORS=$1

tmux new-session -d -t impala_pong

tmux new-window -d -n learner
COMMAND_LEARNER='python3.7 pong_learner.py --env_num '"${NUM_ACTORS}"''
echo $COMMAND_LEARNER

tmux send-keys -t "learner" "$COMMAND_LEARNER" ENTER

sleep 2

for ((id=0; id < $NUM_ACTORS; id++)); do
    tmux new-window -d -n "actor_${id}"
    COMMAND='python3.7 pong_actor.py --env_id  '"${id}"''
    tmux send-keys -t "actor_${id}" "$COMMAND" ENTER

    sleep 0.5
done

tmux attach -t impala_pong