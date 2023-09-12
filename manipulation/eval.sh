python ./test_manipulation.py \
--data_split ./mobility_dataset \
--test_data ./test_data \
--mode manipulation \
--model_path ./ckpts/model_best.pth \
--cfg_file ./config/config.yaml \
--save_path ./exp
###
# --data_split:  your path to mobility_dataset
# --test_data:   your path to test data
# --mode:        manipulation or exploration
# --model_path:  your path to pretrained model
# --cfg_file:    your path to config
# --save_path:   the path to save results of manipulation 
###