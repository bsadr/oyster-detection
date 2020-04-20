for n in `ls pids/*`; do
	kill -9 `cat $n`;
done
if [[  -e "pids" ]]; then
    rm -r "pids"
fi