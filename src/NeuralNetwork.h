#pragma once

#include <vector>
#include <math.h>

class Layer {
public:
	int numOfIncomingNodes, numOfNodes;
	std::vector<std::vector<double>> weights;
	std::vector<double> biases;

	Layer(int numOfIncomingNodes, int numOfNodes);
	std::vector<double> FeedForward(std::vector<double>& input, double (*activationFunction)(double));
};

class NeuralNetwork {
public:
	std::vector<Layer> layers;

	NeuralNetwork(std::vector<int> numberOfNeurons);
	std::vector<double> CalculateOutput(std::vector<double> input, double (*activationFunction)(double));
};

double ReLU(double input);
double Sigmoid(double input);
double HyperbolicTangent(double input);