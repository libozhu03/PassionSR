CUDA_VISIBLE_DEVICES="2" \
python measure.py \
    -i results/quantize/saw_sep/UV/U_W8A8_V_W8A8/DIV2K_val \
    -r ../data/DIV2K/DIV2K_valid_HR

CUDA_VISIBLE_DEVICES="3" \
python measure.py \
    -i results/quantize/saw_sep/UV/U_W6A6_V_W6A6/DIV2K_val \
    -r ../data/DIV2K/DIV2K_valid_HR

CUDA_VISIBLE_DEVICES="6" \
python measure.py \
    -i results/quantize/saw_sep/U/W8A8/DIV2K_val \
    -r ../data/DIV2K/DIV2K_valid_HR

CUDA_VISIBLE_DEVICES="6" \
python measure.py \
    -i results/quantize/saw_sep/U/W6A6/DIV2K_val \
    -r ../data/DIV2K/DIV2K_valid_HR