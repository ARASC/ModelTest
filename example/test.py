from modeltest.dataset.criteo import Criteo


if __name__ == "__main__":
    criteo_dataset = Criteo()
    criteo_dataset.download()