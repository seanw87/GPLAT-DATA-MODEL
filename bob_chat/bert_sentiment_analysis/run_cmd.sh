python run_classifier.py  --task_name=Bob  --do_train=True  --do_eval=True  --data_dir=data  --vocab_file=vocab.txt  --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json  --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt  --train_batch_size=32  --learning_rate=5e-5  --num_train_epochs=2.0  --max_seq_length=128  --output_dir=data/weibo_output/

python run_classifier.py  --task_name=Bob  --do_predict=True  --data_dir=data/bob  --vocab_file=vocab.txt  --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json  --max_seq_length=128  --output_dir=data/bob_output/
