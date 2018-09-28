import yaml


class Config(object):

    def __init__(self):
        self.config = {}

    def add(self, name, value):
        self.config[name] = value

    def save(self, path):
        with open(path, mode='w') as f:
            f.write(yaml.dump(self.config, default_flow_style=False))