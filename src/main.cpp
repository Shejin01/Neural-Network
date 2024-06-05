#include <iostream>
#include "NeuralNetwork.h"

int main() {
	NeuralNetwork nn({2, 3, 1});

	double learningRate = 1.5;

	for (int i = 0; i <= 1000; i++) {
		for (int j = 0; j < 4; j++) {
			int a = j & 1;
			int b = (j & 2) >> 1;
			nn.Learn({ (double)b, (double)a }, { (double)(a^b) }, learningRate);
		}
		if (i % 100 == 0) std::cout << "[*] Epoch: " << i << '\n';
	}
	
	std::vector<double> output;
	for (int i = 0; i < 4; i++) {
		int a = i & 1;
		int b = (i & 2) >> 1;
		output.push_back(nn.CalculateOutput({(double)b, (double)a})[0]);
		std::cout << "[*] " << a << ' ' << b << " => " << (output[i] > 0.5 ? 1 : 0)
			<< " Cost: " << nn.Cost({output[i]}, {(double)(a ^ b)}) << '\n';
	}

	return 0;
}