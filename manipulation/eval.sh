python ./test_manipulation.py \
--data_split ./mobility_dataset \# your path to mobility_dataset
--test_data ./test_data \# your path to test data
--mode manipulation \
--model_path ./ckpts/model_best.pth \# your path to pretrained model
--cfg_file ./config/config.yaml \# your path to config
--save_path ./exp# the path to save results of manipulation 
