sims=(/build/release/samples) # bsim ngpu (all sims)
gpus=("0" "0,1" "0,1,2,3" "0,1,2,3,4,5,6,7")
gpu=("single" "multi" "multi" "multi")
sizes=({250000000..3000000000..250000000}
{250000000..750000000..250000000} {1000000000..3500000000..500000000} {4000000000..6000000000..1000000000}
{250000000..750000000..250000000} {1000000000..5000000000..1000000000} {6000000000..12000000000..2000000000}
{250000000..750000000..250000000} 1000000000 2000000000 4000000000 6000000000 {8000000000..24000000000..4000000000})

sizesbp=({50000000..600000000..50000000}
{50000000..350000000..50000000} {400000000..1200000000..200000000}
{50000000..150000000..50000000} {200000000..1000000000..200000000} {1200000000..2400000000..400000000}
{50000000..150000000..50000000} 200000000 400000000 800000000 1200000000 {1600000000..4800000000..800000000})
echo "[" > results.json

run() {
	echo $1
	eval "$1 >> results.json"
	echo -n "," >> results.json
}

for sim in ${sims[@]}
do
	export CUDA_VISIBLE_DEVICES=0
	
	# simtime sparse
	for size in {250000000..3000000000..250000000}
	do
		run ".$sim --bench sim --gpu single --model synth --pconnect 0.00156 --pfire 0.005 --delay 1 --nsyn $size"
	done

	for igpu in {0..3}
	do
		export CUDA_VISIBLE_DEVICES=${gpus[$igpu]}

		# simtime
		for model in vogels brunel
		do
			for size in ${sizes[@]:$igpu*12:12}
			do
				run ".$sim --bench sim --gpu ${gpu[$igpu]} --model $model --nsyn $size"
			done
		done
		for size in ${sizesbp[@]:$igpu*12:12}
		do
			run ".$sim --bench sim --gpu ${gpu[$igpu]} --model brunel+ --nsyn $size"
		done

		# setup time
		for size in ${sizes[@]:$igpu*12:12}
		do
			run ".$sim --bench setup --gpu ${gpu[$igpu]} --model synth --pconnect 0.05 --pfire 0.005 --delay 1 --nsyn $size"
		done
	done
done

# TODO: speedup (as function of delay) (metrics, viz, ..)

echo -n "{}]" >> results.json
