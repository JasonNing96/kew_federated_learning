python flq_fed.py --dataset mnist --mode qgd        --iters 1000 --out_prefix results
python flq_fed.py --dataset mnist --mode laq8       --iters 1000 --out_prefix results
python flq_fed.py --dataset mnist --mode flq_lowbit --iters 1000 --out_prefix results
python flq_fed.py --dataset mnist --mode flq_bin    --iters 1000 --b_up 1 --out_prefix results

python plot_flq_fed.py --excel_dir . --dataset mnist --save --prefix results