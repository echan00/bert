# Google console to start VM and TPU
ctpu up 
or 
ctpu up --preemptible

# check status of TPU
ctpu status

# add google storage bucket name
export STORAGE_BUCKET=gs://sentence-classification

# copy libraries over to the VM
gsutil cp -r gs://sentence-classification/sentence-alignment-classification-model_Feb_11_2019/*.* ./

# training
python3 run_classifier.py \
  --task_name=supe \
  --do_train=true \
  --data_dir=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/Supervised_Data \
  --vocab_file=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/model/multi_cased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/model/multi_cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/model/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --output_dir=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/output_apr_30_2019 \
  --do_lower_case=False \
  --use_tpu=True \
  --tpu_name=erik-chan

# evaluating

gsutil cp -r gs://sentence-classification/sentence-alignment-classification-model_Feb_11_2019/Supervised_Data/dev.tsv ./Supervised_Data

python3 run_classifier.py \
  --task_name=supe \
  --do_tfx_eval=true \
  --data_dir=./Supervised_Data \
  --vocab_file=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/model/multi_cased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/model/multi_cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/model/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=512 \
  --output_dir=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/output_apr_30_2019/ \
  --do_lower_case=False \
  --use_tpu=True \
  --tpu_name=erik-chan  


#### INFERENCE

# add swap memory
sudo dd if=/dev/zero of=/var/swap2 bs=1M count=16384
sudo chmod 600 /var/swap2
sudo mkswap /var/swap2
sudo swapon /var/swap2

# sudo pico /etc/fstab
/var/swap2 none swap sw 0 0
 
# move files around
mkdir /home/erik_chan/6_supervisor-eval
mkdir /home/erik_chan/6_supervisor-eval/results
gsutil cp -m -r gs://sentence-classification/final-dec-14-2018/6_supervisor-eval/*.tsv ./6_supervisor-eval/
gsutil cp -m -r gs://sentence-classification/final-dec-14-2018/6_supervisor-eval/results/*.tsv ./6_supervisor-eval/results

# split files into smaller chunks 100k in bytes
e.g   split -b 100000k test-ISDA-randomized.tsv test-ISDA-randomized.tsv

# takes all files test*.tsv files in data_dir for inference, each file should have lines "0 \t sentence1 \t sentence2"
# inferenence results are in folders '/results/great_align_', '/results/good_align_', '/results/not_align_', '/results/semi_align_'
python3 run_classifier.py \
  --task_name=supe \
  --do_tfx_predict=true \
  --data_dir=/home/erik_chan/6_supervisor-eval/ \
  --vocab_file=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/model/multi_cased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/model/multi_cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/model/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=512 \
  --output_dir=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/output_apr_30_2019/ \
  --do_lower_case=False \
  --use_tpu=True \
  --tpu_name=erik-chan  


# move data to Google Storage and then to Amazon s3
gsutil cp -r ./ gs://sentence-classification/final-dec-14-2018/6_supervisor-eval/results
gsutil -m rsync -r gs://sentence-classification/final-dec-14-2018/6_supervisor-eval/results s3://nda-ai/final-dec-14-2018/6_supervisor-eval/results

# Exit the VM
exit

# DELETE the VM and TPU
ctpu delete



##### EXPORT MODEL FOR SERVING

python3 run_classifier.py \
  --task_name=supe \
  --data_dir=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/Supervised_Data \
  --vocab_file=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/model/multi_cased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/model/multi_cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/model/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=512 \
  --output_dir=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/output_apr_30_2019 \
  --do_lower_case=False \
  --use_tpu=True \
  --tpu_name=erik-chan \
  --do_export=True \
  --export_dir=$STORAGE_BUCKET/sentence-alignment-classification-model_Feb_11_2019/output_apr_30_2019_export_1_2
