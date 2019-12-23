# ifndef _loss_
# define _loss_

# include "tensor.cpp"

Tensor& L1Loss(Tensor& y, Tensor& z) {
	return mean(abs(y - z));
}

Tensor& CrossEntropyLoss(Tensor& y, Tensor& z) {
	return mean(- y * log(z));
}

Tensor& MSELoss(Tensor& y, Tensor& z) {
	return mean((y - z) ^ 2);
}

# endif