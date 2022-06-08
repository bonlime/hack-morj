### Experiments

Gernet L + drop path (had to modify qubvel library in place for that, dirty but works).
`python3 train.py --train_data_dir ~/data/some_data/ --val_data_dir ~/data/some_data --epochs 200 --model_kwargs '{"arch": "FPN", "encoder_name": "tu-gernet_l", "encoder_kwargs": {"drop_path_rate": 0.1}}' --batch_size 32`
