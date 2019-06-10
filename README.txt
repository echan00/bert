
# train.tsv, dev.tsv, test.tsv in data directory
# labels: ["great-aligned", "good-aligned", "not-aligned", "semi-aligned"]

  def get_labels(self):
    """See base class."""
    return ["great-aligned", "good-aligned", "not-aligned", "semi-aligned"]



# TRAINING
# takes train.tsv, dev.tsv in data_dir folder
# in column 1: 0 is great-aligned, 1 is good-aligned, 2 is not-aligned, 3 is semi-aligned
python3 run_classifier.py \
  --task_name=supe \
  --do_train=true \
  --data_dir=/home/eee/sentence-alignment-classification-model/Supervised_Data \
  --vocab_file=/home/eee/sentence-alignment-classification-model/model/multi_cased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=/home/eee/sentence-alignment-classification-model/model/multi_cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=/home/eee/sentence-alignment-classification-model/model/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=12 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/home/eee/sentence-alignment-classification-model/output_apr_25_2019 \
  --do_lower_case=False

# Train using multi-gpu Horovod version of BERT  https://github.com/lambdal/bert
mpirun -np 3 \
    -H localhost:3 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 run_classifier_hvd.py \
    --task_name=supe \
    --do_train=true \
    --do_eval=true \
    --data_dir=/home/eee/sentence-alignment-classification-model/Supervised_Data \
    --vocab_file=/home/eee/sentence-alignment-classification-model/model/multi_cased_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=/home/eee/sentence-alignment-classification-model/model/multi_cased_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=/home/eee/sentence-alignment-classification-model/model/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
    --max_seq_length=156 \
    --train_batch_size=12 \
    --learning_rate=2e-5 \
    --num_train_epochs=3.0 \
    --output_dir=/home/eee/sentence-alignment-classification-model/output_hvd_apr_24_2019_augment \
    --do_lower_case=False



# EVAL
# takes dev.tsv in data_dir
# in column 1: 0 is great-aligned, 1 is good-aligned, 2 is not-aligned, 3 is semi-aligned
python3 run_classifier.py \
  --task_name=supe \
  --do_tfx_eval=true \
  --data_dir=/home/eee/sentence-alignment-classification-model/Supervised_Data \
  --vocab_file=/home/eee/sentence-alignment-classification-model/model/multi_cased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=/home/eee/sentence-alignment-classification-model/model/multi_cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=/home/eee/sentence-alignment-classification-model/model/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=128 \
  --output_dir=/home/eee/sentence-alignment-classification-model/output_hvd_apr_21_2019_augment/ \
  --do_lower_case=False

# INFERENCE
# takes all files test* in data_dir for inference, each file should have lines "0 \t sentence1 \t sentence2"
# inferenence results are in folders '/results/align_', '/results/not_align_', '/results/semi_align_'
python3 run_classifier.py \
  --task_name=supe \
  --do_tfx_predict=true \
  --data_dir=/home/eee/sentence-alignment-classification-model/data \
  --vocab_file=/home/eee/sentence-alignment-classification-model/model/multi_cased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=/home/eee/sentence-alignment-classification-model/model/multi_cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=/home/eee/sentence-alignment-classification-model/model/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=128 \
  --output_dir=/home/eee/sentence-alignment-classification-model/output_apr_20_2019_augment/ \
  --do_lower_case=False
