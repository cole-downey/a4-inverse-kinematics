#include <iostream>
#include <vector>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <Eigen/Dense>

#include "GLSL.h"
#include "Program.h"
#include "MatrixStack.h"
#include "Shape.h"
#include "Link.h"
#include "Texture.h"

#include "Objective.h"
#include "ObjectiveLinks.h"

#include "Optimizer.h"
#include "OptimizerGD.h"
#include "OptimizerGDLS.h"
#include "OptimizerNM.h"
#include "OptimizerHybrid.h"


using namespace std;
using namespace glm;
using namespace Eigen;

bool keyToggles[256] = { false }; // only for English keyboards!

GLFWwindow* window; // Main application window
string RESOURCE_DIR = ""; // Where the resources are loaded from

shared_ptr<Program> progSimple;
shared_ptr<Program> progTex;
shared_ptr<Shape> shape;
shared_ptr<Texture> texture;
shared_ptr<Objective> objective;

int NLINKS = 4;
vector<shared_ptr<Link> > links;
VectorXd linkAngles;
shared_ptr<ObjectiveLinks> obj;
OptimizerGDLS optimizer;
OptimizerNM optimizerNM;

static void error_callback(int error, const char* description) {
	cerr << description << endl;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

static void char_callback(GLFWwindow* window, unsigned int key) {
	keyToggles[key] = !keyToggles[key];
	switch (key) {
	case 'r':
		// Reset all angles to 0.0
		for (int i = 0; i < NLINKS; ++i) {
			linkAngles.setZero();
		}
		links[0]->setAngles(linkAngles);
		break;
	case '.':
		// Increment all angles
		if (!keyToggles[(unsigned)' ']) {
			for (int i = 0; i < NLINKS; ++i) {
				linkAngles(i) = linkAngles(i) + 0.1;
			}
			links[0]->setAngles(linkAngles);
		}
		break;
	case ',':
		// Decrement all angles
		if (!keyToggles[(unsigned)' ']) {
			for (int i = 0; i < NLINKS; ++i) {
				linkAngles(i) = linkAngles(i) - 0.1;
			}
			links[0]->setAngles(linkAngles);
		}
		break;
	case 'p': {

		//obj->setTarget(NLINKS - 1, 1);
		auto p = obj->p(linkAngles);
		auto p1 = obj->p1(linkAngles);
		auto p2 = obj->p2(linkAngles);
		VectorXd g(NLINKS);
		MatrixXd H(NLINKS, NLINKS);
		cout << "P: " << endl << p.segment<2>(0) << endl;
		cout << "P1: " << endl << p1 << endl;
		cout << "P2: " << endl << p2 << endl;
		auto f = obj->evalObjective(linkAngles, g, H);
		cout << "F: " << endl << f << endl;
		cout << "G: " << endl << g << endl;
		cout << "H: " << endl << H << endl;
	} break;
	case 'o': {

		linkAngles = optimizer.optimize(obj, linkAngles);
		links[0]->setAngles(linkAngles);
	} break;
	}
}

static void cursor_position_callback(GLFWwindow* window, double xmouse, double ymouse) {
	// Get current window size.
	int width, height;
	glfwGetWindowSize(window, &width, &height);
	// Convert from window coord to world coord assuming that we're
	// using an orthgraphic projection
	double aspect = (double)width / height;
	double ymax = NLINKS;
	double xmax = aspect * ymax;
	Vector2d x;
	x(0) = 2.0 * xmax * ((xmouse / width) - 0.5);
	x(1) = 2.0 * ymax * (((height - ymouse) / height) - 0.5);
	if (keyToggles[(unsigned)' ']) {
		obj->setTarget(x(0), x(1));
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	// Get the current mouse position.
	double xmouse, ymouse;
	glfwGetCursorPos(window, &xmouse, &ymouse);
}

static void init() {
	GLSL::checkVersion();

	// Set background color
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	// Enable z-buffer test
	glEnable(GL_DEPTH_TEST);

	keyToggles[(unsigned)'c'] = true;

	progSimple = make_shared<Program>();
	progSimple->setShaderNames(RESOURCE_DIR + "simple_vert.glsl", RESOURCE_DIR + "simple_frag.glsl");
	progSimple->setVerbose(true); // Set this to true when debugging.
	progSimple->init();
	progSimple->addUniform("P");
	progSimple->addUniform("MV");
	progSimple->setVerbose(false);

	progTex = make_shared<Program>();
	progTex->setVerbose(true); // Set this to true when debugging.
	progTex->setShaderNames(RESOURCE_DIR + "tex_vert.glsl", RESOURCE_DIR + "tex_frag.glsl");
	progTex->init();
	progTex->addUniform("P");
	progTex->addUniform("MV");
	progTex->addAttribute("aPos");
	progTex->addAttribute("aTex");
	progTex->addUniform("texture0");
	progTex->setVerbose(false);

	texture = make_shared<Texture>();
	texture->setFilename(RESOURCE_DIR + "metal_texture_15_by_wojtar_stock.jpg");
	texture->init();
	texture->setUnit(0);

	shape = make_shared<Shape>();
	shape->loadMesh(RESOURCE_DIR + "link.obj");
	shape->setProgram(progTex);
	shape->init();

	// Initialize time.
	glfwSetTime(0.0);

	// If there were any OpenGL errors, this will print something.
	// You can intersperse this line in your code to find the exact location
	// of your OpenGL error.
	GLSL::checkError(GET_FILE_LINE);

	// Create the links
	Eigen::Matrix4d meshMat = Eigen::Matrix4d::Identity();
	meshMat(0, 3) = 0.5; // offsetting mesh by 0.5, 0
	for (int i = 0; i < NLINKS; i++) {
		auto l = make_shared<Link>();
		l->setMeshMatrix(meshMat);
		links.push_back(l);
		if (i) {
			l->setPosition(1.0, 0.0);
			links.at(i - 1)->addChild(l);
		}
	}

	// initialize angles
	linkAngles = VectorXd(NLINKS);
	linkAngles.setZero();

	obj = make_shared<ObjectiveLinks>();
	obj->addLinks(links);
	obj->setTarget(NLINKS - 1, 1);

	optimizer.setAlphaInit(0.0001);
	optimizer.setIterMax(10 * NLINKS);
	optimizer.setTol(0.001);
	optimizerNM.setIterMax(10 * NLINKS);
	optimizerNM.setTol(0.001);

}

void render() {
	// Get current frame buffer size.
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	glViewport(0, 0, width, height);

	// Clear buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if (keyToggles[(unsigned)'c']) {
		glEnable(GL_CULL_FACE);
	} else {
		glDisable(GL_CULL_FACE);
	}
	if (keyToggles[(unsigned)'l']) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	} else {
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}

	// Optimize angles
	if (keyToggles[(unsigned)' ']) {
		VectorXd gdAngles = optimizer.optimize(obj, linkAngles);
		VectorXd nmAngles = optimizerNM.optimize(obj, gdAngles);
		if (optimizer.getF() < optimizerNM.getF()) {
			linkAngles = gdAngles;
		} else {
			linkAngles = nmAngles;
		}
		links[0]->setAngles(linkAngles);
		if (keyToggles[(unsigned)'i']) {
			cout << "Iterations(GDLS): " << optimizer.getIter() << endl;
			cout << "Iterations(NM): " << optimizerNM.getIter() << endl;
		}
		for (int i = 0; i < NLINKS; i++) {
			while (linkAngles(i) > 3.14159) {
				linkAngles(i) -= 2 * 3.14159;
			}
			while (linkAngles(i) < -3.14159) {
				linkAngles(i) += 2 * 3.14159;
			}
		}

	}


	auto P = make_shared<MatrixStack>();
	auto MV = make_shared<MatrixStack>();
	P->pushMatrix();
	MV->pushMatrix();

	// Apply camera transforms
	double aspect = (double)width / height;
	double ymax = NLINKS;
	double xmax = aspect * ymax;
	P->multMatrix(glm::ortho(-xmax, xmax, -ymax, ymax, -1.0, 1.0));

	// Draw grid
	progSimple->bind();
	glUniformMatrix4fv(progSimple->getUniform("P"), 1, GL_FALSE, glm::value_ptr(P->topMatrix()));
	glUniformMatrix4fv(progSimple->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
	// Draw axes
	glLineWidth(2.0f);
	glColor3d(0.2, 0.2, 0.2);
	glBegin(GL_LINES);
	glVertex2d(-xmax, 0.0);
	glVertex2d(xmax, 0.0);
	glVertex2d(0.0, -ymax);
	glVertex2d(0.0, ymax);
	glEnd();
	// Draw grid lines
	glLineWidth(1.0f);
	glColor3d(0.8, 0.8, 0.8);
	glBegin(GL_LINES);
	for (int x = 1; x < xmax; ++x) {
		glVertex2d(x, -ymax);
		glVertex2d(x, ymax);
		glVertex2d(-x, -ymax);
		glVertex2d(-x, ymax);
	}
	for (int y = 1; y < ymax; ++y) {
		glVertex2d(-xmax, y);
		glVertex2d(xmax, y);
		glVertex2d(-xmax, -y);
		glVertex2d(xmax, -y);
	}
	glEnd();
	progSimple->unbind();

	// Draw shape
	progTex->bind();
	texture->bind(progTex->getUniform("texture0"));
	glUniformMatrix4fv(progTex->getUniform("P"), 1, GL_FALSE, glm::value_ptr(P->topMatrix()));
	MV->pushMatrix();
	// TODO: draw the links recursively
	if (links.size()) links.at(0)->draw(progTex, MV, shape);
	MV->popMatrix();
	texture->unbind();
	progTex->unbind();

	//////////////////////////////////////////////////////
	// Cleanup
	//////////////////////////////////////////////////////

	// Pop stacks
	MV->popMatrix();
	P->popMatrix();

	GLSL::checkError(GET_FILE_LINE);
}

int main(int argc, char** argv) {
	if (argc < 2) {
		cout << "Please specify the resource directory." << endl;
		return 0;
	}
	RESOURCE_DIR = argv[1] + string("/");
	if (argc > 2) {
		NLINKS = atoi(argv[2]);
	}

	// Set error callback.
	glfwSetErrorCallback(error_callback);
	// Initialize the library.
	if (!glfwInit()) {
		return -1;
	}
	// Create a windowed mode window and its OpenGL context.
	window = glfwCreateWindow(640 * 3, 480 * 3, "Cole Downey", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}
	// Make the window's context current.
	glfwMakeContextCurrent(window);
	// Initialize GLEW.
	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		cerr << "Failed to initialize GLEW" << endl;
		return -1;
	}
	glGetError(); // A bug in glewInit() causes an error that we can safely ignore.
	cout << "OpenGL version: " << glGetString(GL_VERSION) << endl;
	cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
	// Set vsync.
	glfwSwapInterval(1);
	// Set keyboard callback.
	glfwSetKeyCallback(window, key_callback);
	// Set char callback.
	glfwSetCharCallback(window, char_callback);
	// Set cursor position callback.
	glfwSetCursorPosCallback(window, cursor_position_callback);
	// Set mouse button callback.
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	// Initialize scene.
	init();
	// Loop until the user closes the window.
	while (!glfwWindowShouldClose(window)) {
		// Render scene.
		render();
		// Swap front and back buffers.
		glfwSwapBuffers(window);
		// Poll for and process events.
		glfwPollEvents();
	}
	// Quit program.
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
