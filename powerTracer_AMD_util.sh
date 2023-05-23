timestamp=$(date +"%Y-%m-%d %H:%M:%S.%3N"); power=$(rocm-smi --csv -P | grep "card0")
echo $timestamp,$power >> $1
