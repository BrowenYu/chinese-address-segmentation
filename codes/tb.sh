# tensorboard --logdir model/pos1103 --host=10.100.2.242 --port=6006
default_port=6006
IP="10.100.2.242"
if [ ! -n "$1" ]
then
    echo "You don't input any model address!"
    exit
else
    echo "You have setting model address $1 , host:${IP}"
fi
if [ ! -n "$2" ]
then
    echo "You don't input any number for port, use default number ${default_port}"
     tensorboard --logdir $1 --host=${IP} --port=${default_port}
else
    echo "You have setting port $1"
     tensorboard --logdir $1 --host=${IP} --port=$2
fi