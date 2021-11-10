#pragma once
#ifndef OBJECTIVE_LINKS_H
#define OBJECTIVE_LINKS_H

#include "Objective.h"

#include <vector>
#include <vector>
#include <memory>
#include "Link.h"

class ObjectiveLinks : public Objective {
public:
	ObjectiveLinks();
	void addLinks(const std::vector<std::shared_ptr<Link> >& _links);
	virtual ~ObjectiveLinks();
	virtual double evalObjective(const Eigen::VectorXd& linkAngles) const;
	virtual double evalObjective(const Eigen::VectorXd& linkAngles, Eigen::VectorXd& g) const;
	virtual double evalObjective(const Eigen::VectorXd& linkAngles, Eigen::VectorXd& g, Eigen::MatrixXd& H) const;

	void setTarget(double x, double y) { target << x, y, 1; };
	Eigen::Vector3d p(const Eigen::VectorXd& linkAngles) const;
	Eigen::MatrixXd p1(const Eigen::VectorXd& linkAngles) const;
	Eigen::MatrixXd p2(const Eigen::VectorXd& linkAngles) const;
private:
	std::vector<std::shared_ptr<Link> > links;
	int NLINKS;
	double wtar;
	double wreg;
	Eigen::Vector3d endEff;
	Eigen::Vector3d target;
	Eigen::VectorXd restTheta;


	Eigen::Matrix3d r(double theta) const;
	Eigen::Matrix3d r1(double theta) const;
	Eigen::Matrix3d r2(double theta) const;

	Eigen::Matrix3d T(int l) const;
};

#endif
