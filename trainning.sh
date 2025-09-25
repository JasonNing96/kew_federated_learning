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

# # QGD (蓝)
# python flq_fed_v2.py --dataset fmnist --mode fedavg --b_down 0 --lr 1e-3 \
#   --up_budget_bits 17000000 --sel_clients 0 --partition non_iid --dir_alpha 0.1 \
#   --thr_scale 0 --warmup 0 --C 1000000000 --M 30

# # LAQ (绿虚线)
# python flq_fed_v2.py --dataset fmnist --mode laq8 --b_down 0 --lr 1e-3 \
#   --up_budget_bits 17000000 --sel_clients 0 --partition non_iid --dir_alpha 0.1 \
#   --thr_scale 0 --warmup 0 --C 1000000000 --M 30

# # FLQ(Ours) = low-bit 上行8b + 下发8b (紫)
# python flq_fed_v2.py --dataset fmnist --mode bbit --b 4 --b_down 8 --lr 1e-3 \
#   --up_budget_bits 17000000 --sel_clients 0 --partition non_iid --dir_alpha 0.1 \
#   --thr_scale 0 --warmup 0 --C 1000000000 --M 30

# QGD = FedAvg：上行32b，下发32b；≈每轮1端
python flq_fed_v3.py --dataset fmnist --mode fedavg --iters 800 --M 30 \
  --batch 64 --lr 5e-4 --cl 5e-4 --seed 42 \
  --partition non_iid --dir_alpha 0.1 \
  --b_down 32 --sel_clients 0 --up_budget_bits 17000000 \
  --thr_scale 0 --warmup 0 --C 1000000000

# LAQ（随机8b）：上行8b，下发32b；≈每轮4端
python flq_fed_v3.py --dataset fmnist --mode laq8 --iters 800 --M 30 \
  --batch 64 --lr 5e-4 --cl 5e-4 --seed 42 \
  --partition non_iid --dir_alpha 0.1 \
  --b_down 32 --sel_clients 0 --up_budget_bits 17000000 \
  --thr_scale 0 --warmup 0 --C 1000000000 

# FLQ(Ours) = bbit：上行8b，下发8b；≈每轮9端
python flq_fed_v3.py --dataset fmnist --mode bbit --b 4 --iters 800 --M 20 \
  --batch 64 --lr 1e-3 --cl 5e-4 --seed 42 \
  --partition non_iid --dir_alpha 0.1 \
  --b_down 8 --sel_clients 0 --up_budget_bits 17000000 \
  --thr_scale 0 --warmup 0 --C 1000000000


# plot result
python plot_flq_fed.py --excel_dir . --dataset fmnist --modes fedavg laq8 bbit --prefix results --max_iter 800 --save

# 
python flq_fed_v3.py --dataset fmnist --partition non_iid --dir_alpha 0.1 \
  --iters 800 --M 20 --batch 64 --lr 0.5e-3 --cl 5e-4 --seed 42 \
  --sel_clients 0 --up_budget_bits 17000000 --thr_scale 0 --warmup 0 --C 1000000000  --mode laq8 --b_down 0
  
# acc = 0.89 loss = 0.28 4/20
  --mode bbit --b 8 --b_down 8
#
  --mode bin --b_down 8 --clip_global 1.0

# acc = 0.86 loss = 0.37 4/20
  --mode bbit --b 8 --b_down 32

# laq: --scale_by_selected 0
  python flq_fed_v3.py --dataset fmnist --partition non_iid --dir_alpha 0.1 \
  --iters 800 --M 20 --batch 64 --lr 0.5e-3 --cl 5e-4 --seed 42 \
  --sel_clients 0 --up_budget_bits 17000000 --thr_scale 0 --warmup 0 --C 1000000000  --mode laq8 --b_down 32 --scale_by_selected 0 --sel_ref 1 --clip_global 0.5
# 
  --mode laq8 --b_down 32

#
  --mode laq8 --b_down 0

# bin binary anlysis
python plot_flq_fed.py --excel_dir . --dataset fmnist --modes bin --save