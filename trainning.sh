# QGD 
# python flq_fed_v2.py --dataset fmnist --mode fedavg --b_down 32 \
#   --lr 5e-4 --sel_clients 0 --up_budget_bits 17000000 \
#   --M 10 --iters 1500 --batch 64 --cl 5e-4 --seed 42 \
#   --partition non_iid --dir_alpha 0.1 --thr_scale 0 --warmup 0 --C 1000000000

#   # laQ
#   python flq_fed_v2.py --dataset fmnist --mode laq8 --b_down 32 \
#   --lr 5e-4 --sel_clients 0 --up_budget_bits 17000000 \
#   --M 10 --iters 1500 --batch 64 --cl 5e-4 --seed 42 \
#   --partition non_iid --dir_alpha 0.1 --thr_scale 0 --warmup 0 --C 1000000000

#   # flq_lowbit
#   python flq_fed_v2.py --dataset fmnist --mode bbit --b 8 --b_down 8 \
#   --lr 5e-4 --sel_clients 0 --up_budget_bits 17000000 \
#   --M 10 --iters 1500 --batch 64 --cl 5e-4 --seed 42 \
#   --partition non_iid --dir_alpha 0.1 --thr_scale 0 --warmup 0 --C 1000000000

#   # flq_bin
#   python flq_fed_v2.py --dataset fmnist --mode bin --b_down 8 \
#   --lr 5e-4 --sel_clients 0 --up_budget_bits 17000000 \
#   --M 10 --iters 1500 --batch 64 --cl 5e-4 --seed 42 \
#   --partition non_iid --dir_alpha 0.1 --thr_scale 0 --warmup 0 --C 1000000000

# QGD (蓝)
python flq_fed_v2.py --dataset fmnist --mode fedavg --b_down 32 --lr 1e-3 \
  --up_budget_bits 17000000 --sel_clients 0 --partition non_iid --dir_alpha 0.1 \
  --thr_scale 0 --warmup 0 --C 1000000000

# LAQ (绿虚线)
python flq_fed_v2.py --dataset fmnist --mode laq8 --b_down 32 --lr 1e-3 \
  --up_budget_bits 17000000 --sel_clients 0 --partition non_iid --dir_alpha 0.1 \
  --thr_scale 0 --warmup 0 --C 1000000000

# FLQ(Ours) = low-bit 上行8b + 下发8b (紫)
python flq_fed_v2.py --dataset fmnist --mode bbit --b 8 --b_down 8 --lr 1e-3 \
  --up_budget_bits 17000000 --sel_clients 0 --partition non_iid --dir_alpha 0.1 \
  --thr_scale 0 --warmup 0 --C 1000000000