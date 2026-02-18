from model import Model


def main():
    if __name__ == "__main__":
        model = Model()
        fight_data, X_train, X_test, y_train, y_test = model.create_training_data()

        model.train_rf(fight_data, X_train, X_test, y_train, y_test)
        model.train_xg(fight_data, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()