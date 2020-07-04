from modeltest.dataset.criteo import Criteo


if __name__ == "__main__":
    criteo_dataset = Criteo()
    path = criteo_dataset.load_data()
    print(path)