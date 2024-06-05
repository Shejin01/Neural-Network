#pragma once

#include <vector>
#include <math.h>
#include <random>
#include <iostream>

class Layer {
public:
	int numOfIncomingNodes, numOfNodes;
	std::vector<std::vector<double>> weights;
	std::vector<std::vector<double>> weightCostGradient;
	std::vector<double> biases;
	std::vector<double> biasCostGradient;
	std::vector<double> weightedValues;
	std::vector<double> activatedValues;
	std::vector<double> nodeValues;

	Layer(int numOfIncomingNodes, int numOfNodes);
	std::vector<double> FeedForward(std::vector<double>& input);
	void UpdateGradient(std::vector<double> inputs);
	void UpdateHiddenLayerNodeValues(Layer& oldLayer);
};

class NeuralNetwork {
private:
	double CostDerivative(double actualOutput, double expectedOutput);
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
double SiLU(double input);
std::vector<double> Softmax(std::vector<double> input);
double SigmoidDerivative(double input);