# ifndef _layer_
# define _layer_

# include<math.h>
# include "tensor.cpp"
# include "functional.cpp"

Tensor& linear(Tensor& t, int input_size, int output_size) {
	double k = sqrt(6.0 / (input_size + output_size));
	int x = t.count;
	graph[x] = Tensor(input_size, output_size);
	graph[x].random_init(-k, k);
	graph[x].is_var();
	graph[x + 1] = Tensor(1, output_size);
	graph[x + 1].random_init(-k, k);
	graph[x + 1].is_var();
	return t % graph[x] + graph[x + 1];
}

Tensor& batchNorm(Tensor& t) {
	int x = t.count;
	graph[x] = Tensor(0); graph[x].is_var();
	graph[x + 1] = Tensor(1); graph[x + 1].is_var(); 
	return graph[x] + graph[x + 1] * (t - mean(t)) / (mean((t - mean(t)) ^ 2) ^ 0.5 + 1e-5);
}

Tensor& lstm(Tensor& t, Tensor& h0, Tensor& c0, Tensor& c1, int input_size, int output_size) {
	int x = t.count;
	double k = sqrt(6.0 / (input_size + output_size));
	graph[x] = Tensor(input_size, output_size); graph[x].random_init(-k, k); graph[x].is_var();
	graph[x + 1] = Tensor(1, output_size); graph[x + 1].random_init(-k, k); graph[x + 1].is_var();
	graph[x + 2] = Tensor(output_size, output_size); graph[x + 2].random_init(-k, k); graph[x + 2].is_var();
	graph[x + 3] = Tensor(1, output_size); graph[x + 3].random_init(-k, k); graph[x + 3].is_var();
	sigmoid(t % graph[x] + graph[x+1] + h0 % graph[x+2] + graph[x+3]);
	
	int i = t.count - 1; x = t.count;
	graph[x] = Tensor(input_size, output_size); graph[x].random_init(-k, k); graph[x].is_var();
	graph[x + 1] = Tensor(1, output_size); graph[x + 1].random_init(-k, k); graph[x + 1].is_var();
	graph[x + 2] = Tensor(output_size, output_size); graph[x + 2].random_init(-k, k); graph[x + 2].is_var();
	graph[x + 3] = Tensor(1, output_size); graph[x + 3].random_init(-k, k); graph[x + 3].is_var();
	sigmoid(t % graph[x] + graph[x+1] + h0 % graph[x+2] + graph[x+3]);

	int f = t.count - 1; x = t.count;
	graph[x] = Tensor(input_size, output_size); graph[x].random_init(-k, k); graph[x].is_var();
	graph[x + 1] = Tensor(1, output_size); graph[x + 1].random_init(-k, k); graph[x + 1].is_var();
	graph[x + 2] = Tensor(output_size, output_size); graph[x + 2].random_init(-k, k); graph[x + 2].is_var();
	graph[x + 3] = Tensor(1, output_size); graph[x + 3].random_init(-k, k); graph[x + 3].is_var();
	tanh(t % graph[x] + graph[x+1] + h0 % graph[x+2] + graph[x+3]);

	int g = t.count - 1; x = t.count;
	graph[x] = Tensor(input_size, output_size); graph[x].random_init(-k, k); graph[x].is_var();
	graph[x + 1] = Tensor(1, output_size); graph[x + 1].random_init(-k, k); graph[x + 1].is_var();
	graph[x + 2] = Tensor(output_size, output_size); graph[x + 2].random_init(-k, k); graph[x + 2].is_var();
	graph[x + 3] = Tensor(1, output_size); graph[x + 3].random_init(-k, k); graph[x + 3].is_var();
	sigmoid(t % graph[x] + graph[x+1] + h0 % graph[x+2] + graph[x+3]);

	int o = t.count - 1; x = t.count;
	c1 = graph[f] * c0 + graph[i] * graph[g];
	return graph[o] * tanh(c1);
}

Tensor& gru(Tensor& t, Tensor& h0, int input_size, int output_size) {
	int x = t.count;
	double k = sqrt(6.0 / (input_size + output_size));
	graph[x] = Tensor(input_size, output_size); graph[x].random_init(-k, k); graph[x].is_var();
	graph[x + 1] = Tensor(1, output_size); graph[x + 1].random_init(-k, k); graph[x + 1].is_var();
	graph[x + 2] = Tensor(output_size, output_size); graph[x + 2].random_init(-k, k); graph[x + 2].is_var();
	graph[x + 3] = Tensor(1, output_size); graph[x + 3].random_init(-k, k); graph[x + 3].is_var();
	sigmoid(t % graph[x] + graph[x+1] + h0 % graph[x+2] + graph[x+3]);

	int r = t.count - 1; x = t.count;
	graph[x] = Tensor(input_size, output_size); graph[x].random_init(-k, k); graph[x].is_var();
	graph[x + 1] = Tensor(1, output_size); graph[x + 1].random_init(-k, k); graph[x + 1].is_var();
	graph[x + 2] = Tensor(output_size, output_size); graph[x + 2].random_init(-k, k); graph[x + 2].is_var();
	graph[x + 3] = Tensor(1, output_size); graph[x + 3].random_init(-k, k); graph[x + 3].is_var();
	sigmoid(t % graph[x] + graph[x+1] + h0 % graph[x+2] + graph[x+3]);

	int z = t.count - 1; x = t.count;
	graph[x] = Tensor(input_size, output_size); graph[x].random_init(-k, k); graph[x].is_var();
	graph[x + 1] = Tensor(1, output_size); graph[x + 1].random_init(-k, k); graph[x + 1].is_var();
	graph[x + 2] = Tensor(output_size, output_size); graph[x + 2].random_init(-k, k); graph[x + 2].is_var();
	graph[x + 3] = Tensor(1, output_size); graph[x + 3].random_init(-k, k); graph[x + 3].is_var();
	tanh(t % graph[x] + graph[x+1] + graph[r] * (h0 % graph[x+2] + graph[x+3]));
	
	int n = t.count - 1;
	return (1 - graph[z]) * graph[n] + graph[z] * h0;
}

# endif
