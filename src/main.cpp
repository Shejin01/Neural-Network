#include <iostream>
#include "NeuralNetwork.h"

int main() {
	NeuralNetwork nn({2, 2, 1});
	std::vector<double> output = nn.CalculateOutput({ 4, 2 }, ReLU);

	for (auto& value : output) {
		std::cout << value << '\n';
	}

	return 0;
}