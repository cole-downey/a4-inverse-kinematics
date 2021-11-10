#include "OptimizerGDLS.h"
#include "Objective.h"
#include <iostream>

using namespace std;
using namespace Eigen;

OptimizerGDLS::OptimizerGDLS() :
	alphaInit(1.0),
	gamma(0.5),
	tol(1e-6),
	iterMax(100),
	iter(0),
	fResult(-1) {

}

OptimizerGDLS::~OptimizerGDLS() {

}

VectorXd OptimizerGDLS::optimize(const shared_ptr<Objective> objective, const VectorXd& xInit) {
	int n = xInit.rows();
	VectorXd x = xInit;
	VectorXd g(n);
	double f;
	for (iter = 1; iter <= iterMax; iter++) {
		f = objective->evalObjective(x, g);
		// line search
		auto alpha = alphaInit;
		VectorXd dx;
		for (int iterLS = 1; iterLS <= iterMax; iterLS++) {
			dx = -alpha * g;
			auto fnew = objective->evalObjective(x + dx); // note: don't update g here
			if (fnew < f) break;
			alpha *= gamma;
		}
		x += dx;
		if (dx.norm() < tol) break;
	}
	fResult = f;
		return x;
}
