exp='vawc_kn_32'

device=0
mkdir exp/$exp
python vaw_main.py --config configs/exp/$exp.yml --device $device --task test

