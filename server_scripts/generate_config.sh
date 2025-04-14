base_dir="/home/zongchen/DeepFeatureIV/server_scripts/configs"
mkdir -p "$base_dir"  # make sure folder exists
outfile_list="/home/zongchen/DeepFeatureIV/server_scripts/all_configs.txt"
> "$outfile_list"     # clear file first

for method in kiv kivadaptband
do
for rho in 0.1 0.25 0.5 0.75 0.9
do
  for size in 1000 5000 10000
  do
    outfile="$base_dir/${size}_${rho}_${method}.json"
    echo '{"n_repeat": 10, "data": {"data_name": "demand", "data_size": '"$size"', "rho": '"$rho"'}, "train_params": {"lam1":[-2, -10], "lam2": [-2, -10], "split_ratio": 0.5}}' > "$outfile"
    echo "${outfile/home\/zongchen/\home/ucabzc9/Scratch}  "$method" " >> "$outfile_list"
  done
done
done
