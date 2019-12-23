# ifndef _functional_
# define _functional_

# include "tensor.cpp"

Tensor& tanh(Tensor &t) {
	return (exp(2 * t) - 1) / (exp(2 * t) + 1);
}

Tensor& hardtanh(Tensor &t, double minval, double maxval) {
	return min(max(t, minval), maxval);
}

Tensor& sigmoid(Tensor &t) {
	return exp(t) / (1 + exp(t));
}

Tensor& relu(Tensor &t) {
	return max(t, 0);
}

Tensor& relu6(Tensor &t) {
	return min(max(t, 0), 6);
}

Tensor& leakyRelu(Tensor &t, double alpha) {
	return max(t, t * alpha);
}

Tensor& leakyRelu(Tensor &t) {
	double alpha = 0.01;
	return max(t, t * alpha);
}

Tensor& pRelu(Tensor& t) {
	int x = t.count;
	graph[x] = Tensor(0.25); graph[x].is_var();
	return max(0, t) + min(0, t * graph[x]);
}

Tensor& selu(Tensor& t) {
	double alpha = 1.6732632423543772848170429916717;
	double scale = 1.0507009873554804934193349852946;
	return scale * (max(0, t) + min(0, alpha * (exp(t) - 1)));
}

Tensor& celu(Tensor& t, double alpha) {
	return max(0, t) + min(0, alpha * (exp(t / alpha) - 1));
}

Tensor& celu(Tensor& t) {
	double alpha = 1;
	return max(0, t) + min(0, alpha * (exp(t / alpha) - 1));
}

Tensor& softmax(Tensor &t) {
	return exp(t) / csum((exp(t)));
}

Tensor& softsign(Tensor &t) {
	return t / (1 + abs(t));
}


# endif