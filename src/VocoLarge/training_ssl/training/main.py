from src.VocoLarge.training_ssl.pipeline.config import Config
from src.VocoLarge.training_ssl.training.engine import run_training


def main():
    args = Config()
    run_training(args)

if __name__ == "__main__":
    main()