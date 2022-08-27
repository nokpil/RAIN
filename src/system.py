# coding=utf-8
class System():
    def __init__(self, name, param_dict):
        self.name = name
        self.param_dict = param_dict
        self.rule_name = 'RAIN_' + str(self.name)
        self.agent_num = param_dict['agent_num']
        self.dt = self.param_dict['dt']
        self.data_step = self.param_dict['data_step']
        self.label_step = self.param_dict['label_step']
        self.state_num = self.param_dict['state_num']
        self.answer_num = self.param_dict['answer_num']
        self.const_num = self.param_dict['const_num']