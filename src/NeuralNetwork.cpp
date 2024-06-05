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
		weightedValues.push_back(output[i]);
		output[i] = Sigmoid(output[i]);
		activatedValues.push_back(output[i]);
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
		cost += pow(actualOutput[i] - expectedOutput[i], 2);
	}
	return cost / actualOutput.size();
}

double NeuralNetwork::CostDerivative(double actualOutput, double expectedOutput) {
	return 2 * (actualOutput - expectedOutput);
}

double SigmoidDerivative(double input) {
	return Sigmoid(input) * (1 - Sigmoid(input));
}

void Layer::UpdateGradient(std::vector<double> inputs) {
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < numOfIncomingNodes; j++) {
			weightCostGradient[i][j] += inputs[j] * nodeValues[i];
		}
		biasCostGradient[i] += nodeValues[i];
	}
}

void Layer::UpdateHiddenLayerNodeValues(Layer& oldLayer) {
	nodeValues = std::vector<double>(numOfNodes, 0);
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < oldLayer.numOfNodes; j++) {
			nodeValues[i] += oldLayer.nodeValues[j] * oldLayer.weights[i][j];
		}
		nodeValues[i] *= SigmoidDerivative(weightedValues[i]);
	}
}

void NeuralNetwork::Learn(std::vector<double> trainingInputData, std::vector<double> expectedOutput, double learningRate) {
	std::vector<double> output = CalculateOutput(trainingInputData);
	
	//std::cout << "A\n";
	Layer& lastLayer = layers[layers.size() - 1];
	for (int i = 0; i < lastLayer.numOfNodes; i++) {
		lastLayer.nodeValues.push_back(CostDerivative(output[i], expectedOutput[i]) * SigmoidDerivative(lastLayer.weightedValues[i]));
	}
	//std::cout << "a\n";
	lastLayer.UpdateGradient(layers[layers.size()-2].activatedValues);

	//std::cout << "B\n";
	for (int i = layers.size() - 2; i > 0; i--) {
		layers[i].UpdateHiddenLayerNodeValues(layers[i + 1]);
		layers[i].UpdateGradient(layers[i-1].activatedValues);
	}

	// Apply All Gradients
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
std::vector<double> Softmax(std::vector<double> input) {
	std::vector<double> activatedOutput;
	double sum = 0;
	for (int i = 0; i < input.size(); i++) {
		sum += exp(input[i]);
	}
	for (int i = 0; i < input.size(); i++) {
		activatedOutput.push_back(exp(input[i]) / sum);
	}
	return activatedOutput;
}