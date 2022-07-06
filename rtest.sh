while true
do
	ls images/angle_snaps/*.jpg |sort -R | tail -$1 |while read file; do
		path=$file
		echo $path
		python3 spectreye.py $path
	done
done
