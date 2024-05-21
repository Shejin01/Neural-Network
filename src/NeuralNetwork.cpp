#include "NeuralNetwork.h"

Layer::Layer(int numOfIncomingNodes, int numOfNodes) : numOfIncomingNodes(numOfIncomingNodes), numOfNodes(numOfNodes) {
	weights = std::vector<std::vector<double>>(numOfNodes, std::vector<double>(numOfIncomingNodes, 1));
	biases = std::vector<double>(numOfNodes, 1);
}

std::vector<double> Layer::FeedForward(std::vector<double>& input, double (*activationFunction)(double)) {
	std::vector<double> output(numOfNodes, 1);
	for (int i = 0; i < numOfNodes; i++) {
		output[i] = biases[i];
		for (int j = 0; j < numOfIncomingNodes; j++) {
			output[i] += input[j] * weights[i][j];
		}
		output[i] = activationFunction(output[i]);
	}
	return output;
}

NeuralNetwork::NeuralNetwork(std::vector<int> numberOfNeurons) {
	for (int i = 0; i < numberOfNeurons.size() - 1; i++) {
		layers.push_back(Layer(numberOfNeurons[i], numberOfNeurons[i + 1]));
	}
}

std::vector<double> NeuralNetwork::CalculateOutput(std::vector<double> input, double (*activationFunction)(double)) {
	for (auto& layer : layers) {
		input = layer.FeedForward(input, activationFunction);
	}
	return input;
}

double ReLU(double input) {
	return fmax(input, 0);
}
double Sigmoid(double input) {
	return 1.0 / (1 + exp(-input));
}