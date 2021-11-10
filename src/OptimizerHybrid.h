#pragma once
#ifndef OPTIMIZER_HYBRID_H
#define OPTIMIZER_HYBRID_H

#include "Optimizer.h"

class Objective;

class OptimizerHybrid : public Optimizer
{
public:
	OptimizerHybrid();
	virtual ~OptimizerHybrid();
	virtual Eigen::VectorXd optimize(const std::shared_ptr<Objective> objective, const Eigen::VectorXd &xInit);
	
	void setAlphaInit(double alphaInit) { this->alphaInit = alphaInit; }
	void setGamma(double gamma) { this->gamma = gamma; }
	void setTol(double tol) { this->tol = tol; }
	void setIterMax(int iterMax) { this->iterMax = iterMax; }
	int getIter() const { return iter; }
	
private:
	double alphaInit;
	double gamma;
	double tol;
	int iterMax;
	int iter;
};

#endif
