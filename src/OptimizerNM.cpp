#include "OptimizerNM.h"
#include "Objective.h"
#include <iostream>

using namespace std;
using namespace Eigen;

OptimizerNM::OptimizerNM() :
	tol(1e-6),
	iterMax(100),
	iter(0),
	fResult(-1) {

}

OptimizerNM::~OptimizerNM() {

}

VectorXd OptimizerNM::optimize(const shared_ptr<Objective> objective, const VectorXd& xInit) {
	int n = xInit.rows();
	VectorXd x = xInit;
	VectorXd g(n);
	MatrixXd H(n, n);
	double f;
	for (iter = 1; iter <= iterMax; iter++) {
		f = objective->evalObjective(x, g, H);
		VectorXd dx = -(H.ldlt().solve(g)); // solving  H x = g
		x += dx;
		if (dx.norm() < tol) break;
	}
	fResult = f;
	return x;
}
