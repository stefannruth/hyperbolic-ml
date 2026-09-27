#include "HMLR.hpp"
#include <iostream>

int main() {
    arma::mat train, test;
    if (!train.load("tests/testset/train.txt", arma::raw_ascii)) {
        std::cerr << "Failed to load testset/train.txt\n";
        return 1;
    }

    if (!test.load("tests/testset/test.txt", arma::raw_ascii)) {
        std::cerr << "Failed to load testset/test.txt\n";
        return 1;
    }
    const size_t features = train.n_cols - 1;
    const double c = 2.0;
    HMLR model(3, features, c);
    model.eta = 0.01;
    model.regresstest(train, 500000);
    arma::vec result(test.n_rows, arma::fill::zeros);
    std::string b{""};
    size_t correct = 0;
    for (size_t row = 0; row < test.n_rows; ++row) {
        arma::vec obs = test.row(row).cols(0, test.n_cols - 2).t();
        size_t prediction = model.predict(obs);
        result(row) = static_cast<double>(prediction);
        if (prediction == static_cast<size_t>(test(row, test.n_cols -1))) {
            b = "true";
            ++correct;
        } else {
            b = "false";
        }
        std::cout <<"For observation " << row << " predict: " << prediction << " | " << b << "\n";
    }
    std::cout << correct << " correct out of " << test.n_rows << " observations \n";
    std::cout << "Training mean cross-entropy: " << model.mean_loss(train) << '\n';
    std::cout << "Test mean cross-entropy: " << model.mean_loss(test) << '\n';
    result.save("result.txt", arma::raw_ascii);
}
