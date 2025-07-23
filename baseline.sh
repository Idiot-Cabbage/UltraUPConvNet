export CUDA_VISIBLE_DEVICES=0

python omni_train.py --output_dir=exp_out/trial_1     --prompt     --base_lr=0.003 --use_pretrained --batch_size=16

#reume
python omni_train.py --output_dir=exp_out/trial_1     --prompt     --base_lr=0.003 --use_pretrained --batch_size=16 --resume=/MICCAI/ours/exp_out/trial_1/latest.pth

python  omni_test.py     --output_dir=exp_out/trial_1     --prompt     --is_saveout