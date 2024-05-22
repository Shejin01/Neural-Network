#include "NeuralNetwork.h"

Layer::Layer(int numOfIncomingNodes, int numOfNodes) : numOfIncomingNodes(numOfIncomingNodes), numOfNodes(numOfNodes) {
	weights = std::vector<std::vector<double>>(numOfNodes, std::vector<double>(numOfIncomingNodes, 0));
	weightCostGradient = weights;
	biases = std::vector<double>(numOfNodes, 1);
	biasCostGradient = biases;
	srand(time(0));
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < numOfIncomingNodes; j++) {
			weights[i][j] = ((double)(rand() % 10000) - 5000.0) / 10000.0;
		}
	}
}

std::vector<double> Layer::FeedForward(std::vector<double>& input) {
	std::vector<double> output(numOfNodes, 0);
	for (int i = 0; i < numOfNodes; i++) {
		output[i] = biases[i];
		for (int j = 0; j < numOfIncomingNodes; j++) {
			output[i] += input[j] * weights[i][j];
		}
		output[i] = Sigmoid(output[i]);
	}
	return output;
}

NeuralNetwork::NeuralNetwork(std::vector<int> numberOfNeurons) {
	for (int i = 0; i < numberOfNeurons.size() - 1; i++) {
		layers.push_back(Layer(numberOfNeurons[i], numberOfNeurons[i + 1]));
	}
}

std::vector<double> NeuralNetwork::CalculateOutput(std::vector<double> input) {
	for (auto& layer : layers) {
		input = layer.FeedForward(input);
	}
	return input;
}

double NeuralNetwork::Cost(std::vector<double> actualOutput, std::vector<double> expectedOutput) {
	double cost = 0;
	for (int i = 0; i < actualOutput.size(); i++) {
		cost += pow(expectedOutput[i] - actualOutput[i], 2);
	}
	return cost / actualOutput.size();
}

void NeuralNetwork::Learn(std::vector<double> trainingInputData, std::vector<double> expectedOutput, double learningRate) {
	const double h = 0.0001;
	double originalCost = Cost(CalculateOutput(trainingInputData), expectedOutput);

	for (auto& layer : layers) {
		for (int i = 0; i < layer.numOfNodes; i++) {
			for (int j = 0; j < layer.numOfIncomingNodes; j++) {
				layer.weights[i][j] += h;
				double newCost = Cost(CalculateOutput(trainingInputData), expectedOutput);
				layer.weights[i][j] -= h;
				layer.weightCostGradient[i][j] = (newCost - originalCost) / h;
			}
		}

		for (int i = 0; i < layer.numOfNodes; i++) {
			layer.biases[i] += h;
			double newCost = Cost(CalculateOutput(trainingInputData), expectedOutput);
			layer.biases[i] -= h;
			layer.biasCostGradient[i] = (newCost - originalCost) / h;
		}
	}
	for (auto& layer : layers) {
		for (int i = 0; i < layer.numOfNodes; i++) {
			for (int j = 0; j < layer.numOfIncomingNodes; j++) {
				layer.weights[i][j] -= layer.weightCostGradient[i][j] * learningRate;
			}
			layer.biases[i] -= layer.biasCostGradient[i] * learningRate;
		}
	}
}

double ReLU(double input) {
	return fmax(input, 0);
}
double Sigmoid(double input) {
	return 1.0 / (1 + exp(-input));
}
double HyperbolicTangent(double input) {
	double e2w = exp(2 * input);
	return (e2w - 1) / (e2w + 1);
}
double SiLU(double input) {
	return input / (1 + exp(-input));
}