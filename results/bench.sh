echo "[" > results.json

run() {
	cmd="./build/release/samples $1"
	echo $cmd
	eval "$cmd >> results.json"
	echo -n "," >> results.json
}

export CUDA_VISIBLE_DEVICES=0

# simtime
for model in vogels brunel
do
	for size in {250000000..3000000000..250000000}
	do
		run "--model $model --nsyn $size"
	done
done
for size in {100000000..800000000..100000000}
do
	run "--model brunel+ --nsyn $size"
done

echo -n "{}]" >> results.json
