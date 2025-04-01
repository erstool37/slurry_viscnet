apt update
apt install tmux
tmux new -s slave1
cd slurry_viscnet
git pull origin main
pip install -r requirements.txt
# tmux attach -t slave1
# vessl storage copy-file volume://vessl-storage/decay-visconly . 