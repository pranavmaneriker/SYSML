market="agora" # agora, sr, sr2, bmr
episode_len=5 # 1, 3, 5
tokenizer="bytebpe_combined_30k" # char_1000, bytebpe_combined_30k
tokenizer_type="bpe" # char, bpe
context="pretrained" # pretrained/train
gpu=1 # 0, 1
data_dir="../data/rasmus/cleaned/splits/$market"

cd ../src

gpu_cmd=""
if [[ $gpu -eq 1 ]]; then 
    gpu_cmd="--gpus 1"
fi
context_cmd=""
if [[ context -eq "pretrained" ]]; then
    context_cmd="--pretrained_context_embedding_path subforum.json"
fi

python -u run.py $gpu_cmd --model_name single --seed 42 --output_dir single_model_output --data_path  $data_dir --tokenizer_path $data_dir/tokenizers/$tokenizer --tokenizer_type $tokenizer_type \
    --max_text_len 256 --episode_len $episode_len --num_workers 4 --batch_size 256 --val_batch_size 256 --model_params_time "emb_dim=64"  --model_params_context "emb_dim=128" --max_epochs 30 --context_tokenizer_path contexts \
    $context_cmd --model_params_combined "model_type='PoolingTransformer'|final_size=32" --model_params_classwise "model_type='sm'" --episode_len $episode_len --train_context True $context_cmd --use_context True --use_time True 

cd ../scripts
