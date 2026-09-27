#include "PCBall.hpp"

PCBall::PCBall(double c) : c(c) {};
double PCBall::lambda(const arma::vec& x) {
    return 2 / (1 - c * arma::dot(x, x));
}

// mobius add
arma::vec PCBall::madd(const arma::vec& x, const arma::vec& y) {
    double s1 = 1 + 2 * c * dot(x,y);
    double s2 = 1 - c * arma::dot(x,x);
    return ((s1 + c * arma::dot(y,y)) * x + s2 * y) / (s1 + c * c * arma::dot(x,x) * arma::dot(y,y));
}

//mobius sub
arma::vec PCBall::msub(const arma::vec& x, const arma::vec& y) {
    arma::vec temp = -1 * y;
    return madd(x, temp);
}

// mobius multiply by scalar
arma::vec PCBall::mmul(const arma::vec& x, const double r) {
    if (arma::all( x == 0.0)) {
        return x;
    }
    double root = sqrt(c);
    double temp = (std::tanh(r * std::atanh(std::min(root * arma::norm(x,2), 1 - epsilon)))) / ( root * arma::norm(x,2));
    return temp * x;
}

// PCBall distance
double PCBall::dist(const arma::vec& x, const arma::vec& y) {
    arma::vec temp = -1 * x;
    return (2 * std::atanh(std::min(sqrt(c) * arma::norm(madd(temp, y), 2), 1 - epsilon))) / sqrt(c);
}

// mobius exponential with base x and parameter c
arma::vec PCBall::exp(const arma::vec& v, const arma::vec& x) {
    double norm = arma::norm(v,2);
    if (norm == 0) {
        return x;
    }
    arma::vec temp = ((std::tanh(sqrt(c) * lambda(x) * norm / 2))/ (sqrt(c) * norm)) * v;
    return madd(x, temp);
}

// mobius log with base x and parameter c
arma::vec PCBall::log(const arma::vec& y, const arma::vec& x) {
    if (arma::all(x == y)) {
        return arma::vec(x.n_rows, arma::fill::zeros);
    }
    arma::vec temp = -1 * x;
    temp = madd(temp, y);
    double tnorm = arma::norm(temp,2);
    if (tnorm == 0) {
        return arma::vec(x.n_rows, arma::fill::zeros);
    }
    return ((2 * std::atanh(std::min(1 - epsilon, sqrt(c) * tnorm)))/ (sqrt(c) * lambda(x) * tnorm)) * temp;
}

arma::vec PCBall::exp0(const arma::vec& v) {
    if (arma::all( v == 0.0)) {
        return v;
    }
    return (std::tanh(sqrt(c) * arma::norm(v)) / (sqrt(c) * arma::norm(v,2))) * v;
}

arma::vec PCBall::log0(const arma::vec& y) {
    if (arma::all( y == 0.0)) {
        return y;
    }
    return (std::atanh(std::min(1 - epsilon, sqrt(c) * arma::norm(y,2))) / (sqrt(c) * arma::norm(y,2))) * y;
}

// parallel transport from 0 to x
arma::vec PCBall::ptransport(const arma::vec& v, const arma::vec& x) {
    return ( 2/ (lambda(x))) * v;
}

// mobius matrix multiplication
arma::vec PCBall::mmatmul(const arma::mat& M, const arma::vec& x) {
    arma::vec y = M * x;
    if (arma::norm(y,2) == 0) {
        return arma::vec(y.n_rows, arma::fill::zeros);
    }
    return (tanh((arma::norm(y, 2) * std::atanh(std::min(1 - epsilon, sqrt(c)*arma::norm(x,2))) / arma::norm(x,2)))/ (sqrt(c) * arma::norm(y,2))) * y;
}

// mobius translation
arma::vec PCBall::mtranslate(const arma::vec& x, const arma::vec& b) {
    return madd(x,b);
}
