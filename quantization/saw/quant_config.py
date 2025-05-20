class Quantize_config():
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.user = config_dict['user'] # weight or act or bias
        self.quant_bits = config_dict['quant_bits']
        self.sign = config_dict['sign']
        self.sym = config_dict["sym"]
        self.split = config_dict['split']
        self.layer_type = config_dict["layer_type"] # 2Dquant or normal_quant
        if self.sym:
            self.sign = True


class QuantizeModel_config():
    def __init__(self, quantize_config: dict):
        self.quantize_config = quantize_config
        self.quantype = quantize_config['quantype']
        self.method = quantize_config['method']
        self.only_weight = quantize_config['only_weight']
        self.split = quantize_config['split']
        if "cali_batch_size" in quantize_config.keys():
            self.cali_batch_size = quantize_config['cali_batch_size']
            self.cali_learning_rate = quantize_config['cali_learning_rate']
            self.cali_epochs = quantize_config['cali_epochs']
        if "output_modelpath" in quantize_config.keys():
            self.output_modelpath = quantize_config['output_modelpath']
        self.layer_type = quantize_config["layer_type"] # 2Dquant or normal_quant
        if "s_alpha" in quantize_config.keys():
            self.s_alpha = quantize_config["s_alpha"]

        weight_config = {'user': 'weight'}
        act_config = {'user': 'act'}

        for key in self.quantize_config.keys():
            if key[0:3] == 'wei':
                weight_config[key[7:]] = self.quantize_config[key]

            elif key[0:3] == 'act':
                act_config[key[4:]] = self.quantize_config[key]

            else:
                weight_config[key] = self.quantize_config[key]
                act_config[key] = self.quantize_config[key]

        self.weight_config = Quantize_config(weight_config)
        self.act_config = Quantize_config(act_config)

