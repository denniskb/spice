sims=(/build/release/samples) # bsim ngpu (all sims)
models=(vogels brunel)
sizes_single=({200000000..3000000000..200000000})
sizes_multi=({500000000..24000000000..500000000})

echo "[" > results.txt

run() {
	echo $1
	eval "$1 >> results.txt"
	echo -n "," >> results.txt
}

for sim in ${sims[@]}
do
	export CUDA_VISIBLE_DEVICES=0
	
	# simtime single-gpu
	for model in vogels brunel
	do
		for size in ${sizes_single[@]}
		do
			run ".$sim --bench sim --gpu single --model $model --nsyn $size"
		done
	done
	for size in {50000000..1000000000..50000000}
	do
		run ".$sim --bench sim --gpu single --model brunel+ --nsyn $size"
	done
	
	# simtime sparse
	for size in ${sizes_single[@]}
	do
		run ".$sim --bench sim --gpu single --model synth --pconnect 0.00155 --pfire 0.01 --delay 1 --nsyn $size"
	done
	
	# setup time
	for size in ${sizes_multi[@]}
	do
		run ".$sim --bench setup --gpu single --model synth --pconnect 0.05 --pfire 0 --delay 1 --nsyn $size"
	done
	
	unset CUDA_VISIBLE_DEVICES
	
	for size in ${sizes_multi[@]}
	do
		run ".$sim --bench setup --gpu multi --model synth --pconnect 0.05 --pfire 0 --delay 1 --nsyn $size"
	done
	
	# simtime multi-gpu
	for gpu in "0,1" "0,1,2,3" "0,1,2,3,4,5,6,7"
	do
		export CUDA_VISIBLE_DEVICES=$gpu
		
		for model in vogels brunel
		do
			for size in ${sizes_multi[@]}
			do
				run ".$sim --bench sim --gpu multi --model $model --nsyn $size"
			done
		done
		for size in {100000000..4000000000..100000000}
		do
			run ".$sim --bench sim --gpu multi --model brunel+ --nsyn $size"
		done
	done	
done

#speedup = our sim x synth model x 3 network sizes x 1..8 delays (8 gpus)
export CUDA_VISIBLE_DEVICES=0
for size in 500000000 1000000000 2000000000
do
	for delay in {1..8}
	do
		run ".${sims[0]} --bench sim --gpu single --model synth --pconnect 0.05 --pfire 0.01 --delay $delay --nsyn $size"
	done
done

unset CUDA_VISIBLE_DEVICES
for size in 4000000000 8000000000 16000000000
do
	for delay in {1..8}
	do
		run ".${sims[0]} --bench sim --gpu multi --model synth --pconnect 0.05 --pfire 0.01 --delay $delay --nsyn $size"
	done
done

echo -n "{}]" >> results.txt
