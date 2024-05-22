#pragma once

#include <iostream>
#include <vector>
#include <math.h>
#include <random>

class Layer {
public:
	int numOfIncomingNodes, numOfNodes;
	std::vector<std::vector<double>> weights;
	std::vector<std::vector<double>> weightCostGradient;
	std::vector<double> biases;
	std::vector<double> biasCostGradient;

	Layer(int numOfIncomingNodes, int numOfNodes);
	std::vector<double> FeedForward(std::vector<double>& input);
};

class NeuralNetwork {
public:
	std::vector<Layer> layers;

	NeuralNetwork(std::vector<int> numberOfNeurons);
	std::vector<double> CalculateOutput(std::vector<double> input);
	double Cost(std::vector<double> actualOutput, std::vector<double> expectedOutput);
	void Learn(std::vector<double> trainingData, std::vector<double> expectedOutput, double learningRate);
};

double ReLU(double input);
double Sigmoid(double input);
double HyperbolicTangent(double input);