





python main_online.py --base_dir /root/autodl-tmp/segment/database/step1_ch3 --in_channels 3 --decoder_fusion True --conv_encoder efficientnet-b4 --attention_encoder mit_b2 --logdir /root/autodl-tmp/segment/outputs/se/step1_ch3 --attention_type se   --data_type tiff
python main_online.py --base_dir /root/autodl-tmp/segment/database/step1_ch7 --in_channels 7 --decoder_fusion True --conv_encoder efficientnet-b4 --attention_encoder mit_b2 --logdir /root/autodl-tmp/segment/outputs/se/step1_ch7 --attention_type se
python main_online.py --base_dir /root/autodl-tmp/segment/database/step2_ch3 --in_channels 3 --decoder_fusion True --conv_encoder efficientnet-b4 --attention_encoder mit_b2 --logdir /root/autodl-tmp/segment/outputs/se/step2_ch3 --attention_type se  --data_type tiff
python main_online.py --base_dir /root/autodl-tmp/segment/database/step2_ch5 --in_channels 5 --decoder_fusion True --conv_encoder efficientnet-b4 --attention_encoder mit_b2 --logdir /root/autodl-tmp/segment/outputs/se/step2_ch5 --attention_type se
python main_online.py --base_dir /root/autodl-tmp/segment/database/step3_ch3 --in_channels 3 --decoder_fusion True --conv_encoder efficientnet-b4 --attention_encoder mit_b2 --logdir /root/autodl-tmp/segment/outputs/se/step3_ch3 --attention_type se -
python main_online.py --base_dir /root/autodl-tmp/segment/database/step3_ch5 --in_channels 5 --decoder_fusion True --conv_encoder efficientnet-b4 --attention_encoder mit_b2 --logdir /root/autodl-tmp/segment/outputs/se/step3_ch5 --attention_type se
python main_online.py --base_dir /root/autodl-tmp/segment/database/step1_ch5 --in_channels 5 --decoder_fusion True --conv_encoder efficientnet-b4 --attention_encoder mit_b2 --logdir /root/autodl-tmp/segment/outputs/se/step1_ch5 --fusion_module mutual_attention

python main_online.py --base_dir /root/autodl-tmp/segment/database/step1_ch5 --in_channels 5 --decoder_fusion True --conv_encoder efficientnet-b4 --attention_encoder mit_b2 --logdir /root/autodl-tmp/segment/outputs/mu/step1_ch5  --gated