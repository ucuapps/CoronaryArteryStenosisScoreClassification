import yaml

from trainer import Trainer

if __name__ == "__main__":

    with open('config.yaml', 'r') as f:
       config = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(config)
    trainer.run()
