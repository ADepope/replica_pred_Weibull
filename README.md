Run on the HPC system would typically be executed as follows:

ml python3
lscpu

export OMP_NUM_THREADS=4

time srun python3 {src_location}/replica_Weibull.py \
--mu 0 --alpha 1.9276483948691618 --ratio 1 --sigma 0.0005 --n 1000 --la 1 \
--num_samples 25000 --q-init-factor 0.95 --num-workers 4 --out-dir {out_dir}
