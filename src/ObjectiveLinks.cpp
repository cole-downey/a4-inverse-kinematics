#include "ObjectiveLinks.h"
#include <cmath>

#include <iostream>

using namespace std;
using namespace Eigen;


ObjectiveLinks::ObjectiveLinks() {
	NLINKS = 0;
	endEff << 1.0, 0.0, 1.0;
	target << 1.0, 0.0, 1.0;
	wtar = 1000;
	wreg = 1;
}

void ObjectiveLinks::addLinks(const vector<shared_ptr<Link> >& _links) {
	links = _links;
	NLINKS = (int)links.size();
	target << NLINKS, 0, 1;
	restTheta = VectorXd(NLINKS);
	restTheta.setZero();
}

ObjectiveLinks::~ObjectiveLinks() {

}

double ObjectiveLinks::evalObjective(const VectorXd& linkAngles) const {
	// f(x)
	double f = 0;
	Vector3d P = p(linkAngles);
	Vector2d deltaP = (P - target).segment<2>(0); // take out homogeneous coord?
	VectorXd deltaTheta = linkAngles - restTheta;
	f += 0.5 * wtar * deltaP.transpose() * deltaP;
	f += 0.5 * wreg * deltaTheta.transpose() * deltaTheta;
	return f;
}

double ObjectiveLinks::evalObjective(const VectorXd& linkAngles, VectorXd& g) const {
	// f(x)
	double f = 0;
	Vector3d P = p(linkAngles);
	Vector2d deltaP = (P - target).segment<2>(0);
	VectorXd deltaTheta = linkAngles - restTheta;
	f += 0.5 * wtar * deltaP.transpose() * deltaP;
	f += 0.5 * wreg * deltaTheta.transpose() * deltaTheta;
	// g(x)
	MatrixXd P1 = p1(linkAngles);
	g = VectorXd(NLINKS);
	g.setZero();
	g = g + wtar * (deltaP.transpose() * P1).transpose();
	g = g + wreg * deltaTheta;
	return f;
}

double ObjectiveLinks::evalObjective(const VectorXd& linkAngles, VectorXd& g, MatrixXd& H) const {
	// f(x)
	double f = 0;
	Vector3d P = p(linkAngles);
	Vector2d deltaP = (P - target).segment<2>(0);
	VectorXd deltaTheta = linkAngles - restTheta;
	f += 0.5 * wtar * deltaP.transpose() * deltaP;
	f += 0.5 * wreg * deltaTheta.transpose() * deltaTheta;
	// g(x)
	MatrixXd P1 = p1(linkAngles);
	g = VectorXd(NLINKS);
	g.setZero();
	g = g + wtar * (deltaP.transpose() * P1).transpose();
	g = g + wreg * deltaTheta;
	// H(x)
	MatrixXd P2 = p2(linkAngles);
	H = MatrixXd(NLINKS, NLINKS);
	H.setZero();
	MatrixXd vecDot(NLINKS, NLINKS); // deltaP dotted with each vector element of P"
	for (int iPartial = 0; iPartial < NLINKS; iPartial++) {
		for (int jPartial = 0; jPartial < NLINKS; jPartial++) {
			vecDot(jPartial, iPartial) = deltaP.transpose() * P2.block<2, 1>(jPartial * 2, iPartial);
		}
	}
	H = H + wtar * (P1.transpose() * P1 + vecDot);
	MatrixXd identXd(NLINKS, NLINKS);
	identXd.setIdentity();
	H = H + wreg * identXd;
	return f;
}

Vector3d ObjectiveLinks::p(const VectorXd& linkAngles) const {
	Matrix3d E = Matrix3d::Identity();
	for (int i = 0; i < NLINKS; i++) {
		E = E * T(i) * r(linkAngles(i));
	}
	return E * endEff;
}

MatrixXd ObjectiveLinks::p1(const VectorXd& linkAngles) const {
	MatrixXd P1 = MatrixXd(2, NLINKS);
	P1.setZero();
	for (int iPartial = 0; iPartial < NLINKS; iPartial++) {
		Matrix3d E = Matrix3d::Identity();
		for (int i = 0; i < NLINKS; i++) {
			if (i == iPartial) { // use r'
				E = E * T(i) * r1(linkAngles(i));
			} else { // use r
				E = E * T(i) * r(linkAngles(i));
			}
		}
		Vector3d pPartial = E * endEff;
		P1.block<2, 1>(0, iPartial) = pPartial.segment<2>(0);
	}
	return P1;
}

MatrixXd ObjectiveLinks::p2(const VectorXd& linkAngles) const {
	MatrixXd P2 = MatrixXd(2 * NLINKS, NLINKS);
	P2.setZero();
	for (int iPartial = 0; iPartial < NLINKS; iPartial++) {
		for (int jPartial = 0; jPartial < NLINKS; jPartial++) {
			Matrix3d E = Matrix3d::Identity();
			for (int i = 0; i < NLINKS; i++) {
				if (i == iPartial && i == jPartial) { // use r''
					E = E * T(i) * r2(linkAngles(i));
				} else if (i == iPartial || i == jPartial) { // use r'
					E = E * T(i) * r1(linkAngles(i));
				} else { // use r
					E = E * T(i) * r(linkAngles(i));
				}
			}
			Vector3d pPartial = E * endEff;
			P2.block<2, 1>(jPartial * 2, iPartial) = pPartial.segment<2>(0);
		}
	}
	return P2;
}

Matrix3d ObjectiveLinks::r(double theta) const {
	Matrix3d R;
	R << cos(theta), -sin(theta), 0,
		sin(theta), cos(theta), 0,
		0, 0, 1;
	return R;
}

Matrix3d ObjectiveLinks::r1(double theta) const {
	Matrix3d R1;
	R1 << -sin(theta), -cos(theta), 0,
		cos(theta), -sin(theta), 0,
		0, 0, 0;
	return R1;
}

Matrix3d ObjectiveLinks::r2(double theta) const {
	Matrix3d R2;
	R2 << -cos(theta), sin(theta), 0,
		-sin(theta), -cos(theta), 0,
		0, 0, 0;
	return R2;
}

Matrix3d ObjectiveLinks::T(int i) const {
	Vector2d linkPos = links.at(i)->getPosition();
	Matrix3d t = Matrix3d::Identity();
	t(0, 2) = linkPos(0);
	t(1, 2) = linkPos(1);
	return t;
}
