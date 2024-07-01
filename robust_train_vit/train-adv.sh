
 ### different choice here：  48
    ## 1. embedding: 1） img_input; 2) conv_input   2
    ## 2. Trans_block: 1) ori_att 2) MLP_att    2
    ## 3. MLP 1) ori_MLP 2） conv_MLP 3) no_MLP   3
    ## 4. norm: 1) LN 2) none         2
    ## 5. skipconnect: 1) residual 2） none    2




## adv train


python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP ori_MLP --norm LN --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP ori_MLP --norm LN --skipconnect none --output_dir

python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP ori_MLP --norm none --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP ori_MLP --norm none --skipconnect none --output_dir

python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP conv_MLP --norm LN --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP conv_MLP --norm LN --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP conv_MLP --norm none --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP conv_MLP --norm none --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP no_MLP --norm LN --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP no_MLP --norm LN --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP no_MLP --norm none --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP no_MLP --norm none --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP ori_MLP --norm LN --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP ori_MLP --norm LN --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP ori_MLP --norm none --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP ori_MLP --norm none --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP conv_MLP --norm LN --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP conv_MLP --norm LN --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP conv_MLP --norm none --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP conv_MLP --norm none --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP no_MLP --norm LN --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP no_MLP --norm LN --skipconnect none --output_dir

python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP no_MLP --norm none --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP no_MLP --norm none --skipconnect none --output_dir

## adv conv input

python main_adv_train.py --embedding conv_input --Trans_block ori_att --MLP ori_MLP --norm LN --skipconnect residual --output_dir
python main_adv_train.py --embedding conv_input --Trans_block ori_att --MLP ori_MLP --norm LN --skipconnect none--output_dir

python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP ori_MLP --norm none --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP ori_MLP --norm none --skipconnect none --output_dir

python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP conv_MLP --norm LN --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP conv_MLP --norm LN --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP conv_MLP --norm none --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP conv_MLP --norm none --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP no_MLP --norm LN --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP no_MLP --norm LN --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP no_MLP --norm none --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block ori_att --MLP no_MLP --norm none --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP ori_MLP --norm LN --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP ori_MLP --norm LN --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP ori_MLP --norm none --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP ori_MLP --norm none --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP conv_MLP --norm LN --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP conv_MLP --norm LN --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP conv_MLP --norm none --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP conv_MLP --norm none --skipconnect none --output_dir


python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP no_MLP --norm LN --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP no_MLP --norm LN --skipconnect none --output_dir

python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP no_MLP --norm none --skipconnect residual --output_dir
python main_adv_train.py --embedding img_input --Trans_block MLP_att --MLP no_MLP --norm none --skipconnect none --output_dir