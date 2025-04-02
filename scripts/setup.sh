apt update
apt install tmux
cd slurry_viscnet
pip install -r requirements.txt
git pull origin main
# vessl storage copy-file volume://vessl-storage/decay-visconly . 
tmux new -s slave1
# tmux attach -t slave1