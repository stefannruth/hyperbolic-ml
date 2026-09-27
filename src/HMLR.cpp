#include "HMLR.hpp"

HMLR::HMLR(size_t k, size_t x, double c) : classes(k), features(x),
    p(x, k, arma::fill::zeros), a(x, k, arma::fill::randn), ball(c) {};

double HMLR::score(const arma::vec& a, const arma::vec& p, const arma::vec& x) {
    arma::vec np = -1 * p;
    arma::vec temp = ball.madd(np, x);
    double pow = (2 * arma::norm(a, 2) * std::asinh(
                (2 * sqrt(ball.c) * arma::dot(temp, a))/ ((1 - ball.c * arma::dot(temp, temp)) * arma::norm(a, 2))
                )) / sqrt(ball.c);
    return pow;
}

// calculates classification probabilities given an input vector obs
// we append the max score and denom to the back of the vector for use
// in the regress_helper function to avoid numerical underflow with std::log
std::vector<double> HMLR::prob(const arma::vec& obs) {
    double max = -std::numeric_limits<double>::infinity();
    std::vector<double> powers;
    powers.reserve(a.n_cols);
    for (size_t x = 0; x < a.n_cols; ++x) {
        double temp = score(a.col(x), p.col(x), obs);
        max = std::max(max, temp);
        powers.push_back(temp);
    }
    double denom = 0.0;
    std::vector<double> result;
    result.reserve(a.n_cols + 2);
    for (const auto& x : powers) {
        auto temp = std::exp(x - max);
        result.push_back(temp);
        denom += temp;
    }
    std::transform(std::ranges::begin(result), std::ranges::end(result), std::ranges::begin(result),
            [&denom] (double& a)
            {return a / denom ;});
    result.push_back(max);
    result.push_back(denom);
    return result;
}

void HMLR::regress(const arma::mat& data, size_t max_iter, double threshold) {
    double prev_j = mean_loss(data);
    double delta = 0.0;
    size_t iterations = 0;

    while (iterations < max_iter) {
        regress_helper(data);
        ++iterations;
        const double curr_j = mean_loss(data);
        delta = curr_j - prev_j;
        prev_j = curr_j;
        if (std::abs(delta) <= threshold) {
            break;
        }
    }
}

// helper function iteratively called inside regress
// returns J the mean cross entropy per observation
double HMLR::regress_helper(const arma::mat& data) {
    arma::mat p_update(p.n_rows, p.n_cols, arma::fill::zeros);
    arma::mat a_update(a.n_rows, a.n_cols, arma::fill::zeros);
    const double divide = 1.0 / data.n_rows;
    const double R = (1 - epsilon) / sqrt(ball.c);
    double J = 0.0;

    for (size_t row = 0; row < data.n_rows; ++row) {
        arma::vec obs = data.row(row).cols(0, data.n_cols - 2).t();
        std::vector<double> probs = prob(obs);
        size_t k = static_cast<size_t>(data(row, data.n_cols - 1));
        double s = score(a.col(k), p.col(k), obs);
        double Z = probs[probs.size() - 1];
        double m = probs[probs.size() - 2];
        J += divide * (std::log(Z) - (s - m));
        probs[k] -= 1;
        for (size_t col = 0; col < p.n_cols; ++col) {
            p_update.col(col) += divide * probs[col] * score_grad_p(p.col(col), a.col(col), obs);
            a_update.col(col) += divide * probs[col] * score_grad_a(p.col(col), a.col(col), obs);
        }
    }
    for (size_t col = 0; col < p.n_cols; ++col) {
        a.col(col) -= eta * a_update.col(col);
        double lambda = ball.lambda(p.col(col));
        arma::vec power = (-eta / (lambda * lambda)) * p_update.col(col);
        arma::vec p_cand = ball.exp(power, p.col(col));
        if (p_cand.is_finite()) {
            p.col(col) = p_cand;
        }
        double norm = arma::norm(p.col(col), 2);
        if (norm > R) {
            p.col(col) *= (R/norm);
        }
    }
    return J;
}

size_t HMLR::predict(const arma::vec& obs) {
    std::vector<double> probs = prob(obs);
    size_t max_pos = 0;
    double max = 0.0;
    for(size_t k = 0; k < probs.size() - 2; ++k) {
        if (max < probs[k]) {
            max = probs[k];
            max_pos = k;
        }
    }
    return max_pos;
}


arma::vec HMLR::score_grad_a(const arma::vec& p, const arma::vec& a, const arma::vec& x) {
    arma::vec np = -p;
    arma::vec z = ball.madd(np, x);
    const double root = sqrt(ball.c);
    const double anorm = arma::norm(a,2);
    const double sanorm = arma::dot(a,a);
    const double t = arma::dot(z,a);
    const double B = 1.0 - ball.c * arma::dot(z,z);
    const double denom = sqrt(B * B * sanorm + 4 * ball.c * t * t);
    return
        (2.0 / (root * anorm)) * std::asinh(2.0 * root * t / (B * anorm)) * a
        + (4.0 / (anorm * denom)) * (sanorm * z - t * a);
}

arma::vec HMLR::score_grad_p(const arma::vec& p, const arma::vec& a, const arma::vec& x) {
    arma::vec np = -1 * p;
    arma::vec z = ball.madd(np, x);
    const double c = ball.c;
    const double anorm = arma::norm(a, 2);
    const double xnorm_sq = arma::dot(x, x);
    const double px = arma::dot(p, x);
    const double t = arma::dot(z,a);
    const double B = 1.0 - c * arma::dot(z,z);
    arma::vec g = (4.0 * anorm / (sqrt(B * B * arma::dot(a,a)+ 4.0 * c * t * t)))
        * (a + (2.0 * c * t /B) * z);
    double A = 1.0 - 2.0 * c * px + c * xnorm_sq;
    double D = 1.0 - 2.0 * c * px + c * c * arma::dot(p,p) * xnorm_sq;
    arma::vec s = -A * g + 2.0 * c * arma::dot(p,g) * x - 2.0 * c * arma::dot(x,g) * p
        + 2.0 * c * arma::dot(z,g) * (x - c * xnorm_sq * p);
    s /= D;
    return s;
}
double HMLR::mean_loss(const arma::mat& data) {
    if (data.n_rows == 0 || data.n_cols != features + 1) {
        throw std::invalid_argument("Invalid dataset dimensions");
    }

    double total = 0.0;

    for (size_t row = 0; row < data.n_rows; ++row) {
        arma::vec obs = data.row(row).cols(0, features - 1).t();

        const double label = data(row, features);
        if (!std::isfinite(label) || label < 0 ||
            label >= classes || label != std::floor(label)) {
            throw std::invalid_argument("Invalid class label");
        }
        const size_t k = static_cast<size_t>(label);

        const auto probs = prob(obs);
        const double m = probs[probs.size() - 2];
        const double Z = probs.back();
        const double s = score(a.col(k), p.col(k), obs);

        total += std::log(Z) - (s - m);
    }

    if (!std::isfinite(total)) {
        throw std::runtime_error("Loss became nonfinite");
    }

    return total / static_cast<double>(data.n_rows);
}
void HMLR::regresstest(
    const arma::mat& data,
    size_t max_iter,
    double threshold
) {
    if (!std::isfinite(threshold) || threshold < 0.0) {
        throw std::invalid_argument("Invalid stopping threshold");
    }
    double prev_j = mean_loss(data);
    double delta = 0.0;
    size_t iterations = 0;
    bool threshold_reached = false;

    while (iterations < max_iter) {
        regress_helper(data);
        ++iterations;
        const double curr_j = mean_loss(data);
        delta = curr_j - prev_j;
        prev_j = curr_j;
        if (std::abs(delta) <= threshold) {
            threshold_reached = true;
            break;
        }
    }

    std::cout << std::setprecision(10) << "Iterations: " << iterations << '\n' << "Stopped because: "
              << (threshold_reached
                      ? "loss-change threshold reached"
                      : "iteration limit reached") << '\n';
    if (iterations > 0) { std::cout << "Final loss change (new - old): " << delta << '\n'; }
}
