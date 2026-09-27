#include "PCBall.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <ranges>
#include <stdexcept>

class HMLR {
    public:
        HMLR(size_t k, size_t x, double c);
        size_t classes;
        size_t features;
        arma::mat p;
        arma::mat a;
        double eta = 0.1;
        double epsilon = 0.00001;
        PCBall ball;
        void regress(const arma::mat& data, size_t max_iter = 1000, double threshold = 0.0000000001);
        size_t predict(const arma::vec& obs);
        void regresstest(const arma::mat& data, size_t max_iter = 1000, double threshold = 0.0000000001);
        double mean_loss(const arma::mat& data);
    private:
        std::vector<double> prob(const arma::vec& obs);
        arma::vec gradient(const arma::vec& x);
        double score(const arma::vec& a, const arma::vec& p, const arma::vec& x);
        arma::vec score_grad_a(const arma::vec& p, const arma::vec& a, const arma::vec& x);
        arma::vec score_grad_p(const arma::vec& p, const arma::vec& a, const arma::vec& x);
        double regress_helper(const arma::mat& data);
};
