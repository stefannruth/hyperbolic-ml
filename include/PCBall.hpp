#include <cmath>
#include <armadillo>

// c defines the radius of the ball 1/sqrt(c)
// n defines the dimension of the ball
class PCBall {
    public:
        PCBall(double c);
        double c;
        double epsilon = 0.00001;
        double lambda(const arma::vec& x);
        arma::vec madd(const arma::vec& x, const arma::vec& y);
        arma::vec msub(const arma::vec& x, const arma::vec& y);
        arma::vec mmul(const arma::vec& x, const double r);
        double dist(const arma::vec& x, const arma::vec& y);
        arma::vec exp(const arma::vec& v, const arma::vec& x);
        arma::vec log(const arma::vec& y, const arma::vec& x);
        arma::vec exp0(const arma::vec& v);
        arma::vec log0(const arma::vec& y);
        arma::vec ptransport(const arma::vec& v, const arma::vec& x);
        arma::vec mmatmul(const arma::mat& M, const arma::vec& x);
        arma::vec mtranslate(const arma::vec& x, const arma::vec& b);
};
