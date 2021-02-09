import torch

train_config={
"hidden_size":300,
"teacher_forcing_ratio":0.5,
"learning_rate":1e-2,
"epochs":10,
"batch_size":32,
"data_path":'/content/drive/MyDrive/coop_data/processed_retail_item_df_with_encoded_breadcrumbs.csv',
"exp_name":"simple_encoder_step_lr_fasttext_augmented",
"exp_name_init":"simple_encoder_step_lr_fasttext",
"device":torch.device("cuda"),
"fasttext_path":"/content/nfs/machine-learning/fasttext/cc.de.300.bin"

}