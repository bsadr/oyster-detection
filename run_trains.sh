if [[ ! -e "logs" ]]; then
    mkdir "logs"
fi
if [[ ! -e "pids" ]]; then
    mkdir "pids"
fi
for i in `seq 0 10`; do
	nohup python oysterd/train.py -i $i > "logs/$i.txt" 2>&1 &
	echo "$!" > "pids/$i.txt";
done
