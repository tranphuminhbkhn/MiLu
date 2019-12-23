# ifndef _tensor_operator_
# define _tensor_operator_

# include <bits/stdc++.h>
# include <mpi.h>
// using namespace std;

MPI_Status status;
int np; int pid;

double **init_array(int rows, int cols) {
	double *data = (double *)malloc(rows*cols*sizeof(double));
	double **array= (double **)malloc(rows*sizeof(double*));
	for (int i=0; i<rows; i++)
		array[i] = &(data[cols*i]);
	return array;
}

void free_array(double ** a) {
	free(a[0]);
	free(a);
}

void add(double ** c, double ** a , int ar, int ac, double ** b, int br, int bc) {
	int n = std :: max(ar, br);
	int m = std :: max(ac, bc);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (ar == br and ac == bc) {
				c[i][j] = a[i][j] + b[i][j];
			}

			else if (ar == 1 and ac == bc) {
				c[i][j] = a[0][j] + b[i][j];
			}

			else if (ar == br and ac == 1) {
				c[i][j] = a[i][0] + b[i][j];
			}

			else if (br == 1 and bc == ac) {
				c[i][j] = a[i][j] + b[0][j]; 
			}

			else if (br == ar and bc == 1) {
				c[i][j] = a[i][j] + b[i][0];
			}

			else if (ar == 1 and ac == 1) {
				c[i][j] = a[0][0] + b[i][j];
			}

			else if (br == 1 and bc == 1) {
				c[i][j] = a[i][j] + b[0][0];
			}
		}
	}
}

void sub(double ** c, double ** a , int ar, int ac, double ** b, int br, int bc) {
	int n = std :: max(ar, br);
	int m = std :: max(ac, bc);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (ar == br and ac == bc) {
				c[i][j] = a[i][j] - b[i][j];
			}

			else if (ar == 1 and ac == bc) {
				c[i][j] = a[0][j] - b[i][j];
			}

			else if (ar == br and ac == 1) {
				c[i][j] = a[i][0] - b[i][j];
			}

			else if (br == 1 and bc == ac) {
				c[i][j] = a[i][j] - b[0][j]; 
			}

			else if (br == ar and bc == 1) {
				c[i][j] = a[i][j] - b[i][0];
			}

			else if (ar == 1 and ac == 1) {
				c[i][j] = a[0][0] - b[i][j];
			}

			else if (br == 1 and bc == 1) {
				c[i][j] = a[i][j] - b[0][0];
			}
		}
	}
}

void mul(double ** c, double ** a , int ar, int ac, double ** b, int br, int bc) {
	int n = std :: max(ar, br);
	int m = std :: max(ac, bc);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (ar == br and ac == bc) {
				c[i][j] = a[i][j] * b[i][j];
			}

			else if (ar == 1 and ac == bc) {
				c[i][j] = a[0][j] * b[i][j];
			}

			else if (ar ==br and ac == 1) {
				c[i][j] = a[i][0] * b[i][j];
			}

			else if (br == 1 and bc == ac) {
				c[i][j] = a[i][j] * b[0][j]; 
			}

			else if (br == ar and bc == 1) {
				c[i][j] = a[i][j] * b[i][0];
			}

			else if (ar == 1 and ac == 1) {
				c[i][j] = a[0][0] * b[i][j];
			}

			else if (br == 1 and bc == 1) {
				c[i][j] = a[i][j] * b[0][0];
			}
		}
	}
}

void div(double ** c, double ** a , int ar, int ac, double ** b, int br, int bc) {
	int n = std :: max(ar, br);
	int m = std :: max(ac, bc);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (ar == br and ac == bc) {
				c[i][j] = a[i][j] / b[i][j];
			}

			else if (ar == 1 and ac == bc) {
				c[i][j] = a[0][j] / b[i][j];
			}

			else if (ar == br and ac == 1) {
				c[i][j] = a[i][0] / b[i][j];
			}

			else if (br == 1 and bc == ac) {
				c[i][j] = a[i][j] / b[0][j]; 
			}

			else if (br == ar and bc == 1) {
				c[i][j] = a[i][j] / b[i][0];
			}

			else if (ar == 1 and ac == 1) {
				c[i][j] = a[0][0] / b[i][j];
			}

			else if (br == 1 and bc == 1) {
				c[i][j] = a[i][j] / b[0][0];
			}
		}
	}
}

void pow(double ** c, double ** a , int ar, int ac, double ** b, int br, int bc) {
	int n = std :: max(ar, br);
	int m = std :: max(ac, bc);

	for(int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (ar == br and ac == bc) {
				c[i][j] = pow(a[i][j], b[i][j]);
			}

			else if (ar == 1 and ac == bc) {
				c[i][j] = pow(a[0][j], b[i][j]);
			}

			else if (ar == br and ac == 1) {
				c[i][j] = pow(a[i][0], b[i][j]);
			}

			else if (br == 1 and bc == ac) {
				c[i][j] = pow(a[i][j], b[0][j]); 
			}

			else if (br == ar and bc == 1) {
				c[i][j] = pow(a[i][j], b[i][0]);
			}

			else if (ar == 1 and ac == 1) {
				c[i][j] = pow(a[0][0], b[i][j]);
			}

			else if (br == 1 and bc == 1) {
				c[i][j] = pow(a[i][j], b[0][0]);
			}
		}
	}
}
void matmul(double ** c, double ** a , int ar, int ac, double ** b, int br, int bc) {
	int n = ar;
	int m = ac;
	int p = bc;

	for (int i = 0; i < n; i++) {
		for (int k = 0; k < p; k++) {
			c[i][k] = 0;
		}
	}
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			for(int k = 0; k < p; k++) {
				c[i][k] += a[i][j] * b[j][k];
			}
		}
	}
}

void transpose(double ** c, double ** a, int ar, int ac) {
	for(int i = 0; i < ac; i++) {
		for(int j = 0; j < ar; j++) {
			c[i][j] = a[j][i];
		}
	}
}

void negative(double ** c, double ** a, int ar, int ac) {
	for(int i = 0; i < ar; i++) {
		for(int j = 0; j < ac; j++) {
			c[i][j] = - a[i][j];
		}
	}
}


void fill(double ** a, int ar, int ac, double v) {
	for(int i = 0; i < ar; i++) {
		for(int j = 0; j < ac; j++) {
			a[i][j] = v;
		}
	}
}

void copy(double ** c, double ** a, int ar, int ac) {
	for(int i = 0; i < ar; i++) {
		for(int j = 0; j < ac; j++) {
			c[i][j] = a[i][j];
		}
	}
}


void exp(double ** c, double ** a, int ar, int ac) {
	for(int i = 0; i < ar; i++) {
		for(int j = 0; j < ac; j++) {
			c[i][j] = exp(a[i][j]);
		}
	}
}

void log(double ** c, double ** a, int ar, int ac) {
	for(int i = 0; i < ar; i++) {
		for(int j = 0; j < ac; j++) {
			c[i][j] = log(a[i][j]);
		}
	}
}

void mean(double ** c, double ** a, int ar, int ac) {
	c[0][0] = 0;
	for(int i = 0; i < ar; i++) {
		for(int j = 0; j < ac; j++) {
			c[0][0] += a[i][j];
		}
	}
	c[0][0] /= ar * ac;
}


void rmean(double ** c, double ** a, int ar, int ac) {
	for(int j = 0; j < ac; j++) {
		c[0][j] = 0;
	}	
	for(int i = 0; i < ar; i++) {
		for(int j = 0; j < ac; j++) {
			c[0][j] += a[i][j];
		}
	}
	for(int j = 0; j < ac; j++) {
		c[0][j] /= ar;
	}
}

void cmean(double ** c, double ** a, int ar, int ac) {
	for(int i = 0; i < ar; i++) {
		c[i][0] = 0;
	}
	for(int i = 0; i < ar; i++) {
		for(int j = 0; j < ac; j++) {
			c[i][0] += a[i][j];
		}
		c[i][0] /= ac;
	}
}


void sum(double ** c, double ** a, int ar, int ac) {
	c[0][0] = 0;
	for(int i = 0; i < ar; i++) {
		for(int j = 0; j < ac; j++) {
			c[0][0] += a[i][j];
		}
	}
}


void rsum(double ** c, double ** a, int ar, int ac) {
	for(int j = 0; j < ac; j++) {
		c[0][j] = 0;
	}	
	for(int i = 0; i < ar; i++) {
		for(int j = 0; j < ac; j++) {
			c[0][j] += a[i][j];
		}
	}
}

void csum(double ** c, double ** a, int ar, int ac) {
	for(int i = 0; i < ar; i++) {
		c[i][0] = 0;
	}
	for(int i = 0; i < ar; i++) {
		for(int j = 0; j < ac; j++) {
			c[i][0] += a[i][j];
		}
	}
}

void abs(double ** c, double ** a, int ar, int ac) {
	for(int i = 0; i < ar; i++) {
		for(int j = 0; j < ac; j++) {
			c[i][j] = a[i][j] >= 0 ? a[i][j] : -a[i][j];
		}
	}
}

void sign(double ** c, double ** a, int ar, int ac) {
	for(int i = 0; i < ar; i++) {
		for(int j = 0; j < ac; j++) {
			c[i][j] = a[i][j] >= 0 ? 1 : -1;
		}
	}
}

void maximum(double ** c, double ** a , int ar, int ac, double ** b, int br, int bc, double ** a_mask, double ** b_mask) {
	int n = std :: max(ar, br);
	int m = std :: max(ac, bc);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			if (ar == br and ac == bc) {
				if (a[i][j] >= b[i][j]) {
					c[i][j] = a[i][j];
					a_mask[i][j] = 1;
					b_mask[i][j] = 0;
				}
				else {
					c[i][j] = b[i][j];
					a_mask[i][j] = 0;
					b_mask[i][j] = 1;
				}
			}

			else if (ar == 1 and ac == 1) {
				if (a[0][0] >= b[i][j]) {
					c[i][j] = a[0][0];
					a_mask[i][j] = 1;
					b_mask[i][j] = 0;
				}
				else {
					c[i][j] = b[i][j];
					a_mask[i][j] = 0;
					b_mask[i][j] = 1;
				}
			}

			else if (br == 1 and bc == 1) {
				if (a[i][j] >= b[0][0]) {
					c[i][j] = a[i][j];
					a_mask[i][j] = 1;
					b_mask[i][j] = 0;
				}
				else {
					c[i][j] = b[0][0];
					a_mask[i][j] = 0;
					b_mask[i][j] = 1;
				}
			}
		}
	}
}


void minimum (double ** c, double ** a , int ar, int ac, double ** b, int br, int bc, double ** a_mask, double ** b_mask) {
	int n = std :: max(ar, br);
	int m = std :: max(ac, bc);

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			if (ar == br and ac == bc) {
				if (a[i][j] < b[i][j]) {
					c[i][j] = a[i][j];
					a_mask[i][j] = 1;
					b_mask[i][j] = 0;
				}
				else {
					c[i][j] = b[i][j];
					a_mask[i][j] = 0;
					b_mask[i][j] = 1;
				}
			}

			else if (ar == 1 and ac == 1) {
				if (a[0][0] < b[i][j]) {
					c[i][j] = a[0][0];
					a_mask[i][j] = 1;
					b_mask[i][j] = 0;
				}
				else {
					c[i][j] = b[i][j];
					a_mask[i][j] = 0;
					b_mask[i][j] = 1;
				}
			}

			else if (br == 1 and bc == 1) {
				if (a[i][j] < b[0][0]) {
					c[i][j] = a[i][j];
					a_mask[i][j] = 1;
					b_mask[i][j] = 0;
				}
				else {
					c[i][j] = b[0][0];
					a_mask[i][j] = 0;
					b_mask[i][j] = 1;
				}
			}
		}
	}
}

void add(double ** c, double ** a, int ar, int ac, double v) {
	double ** b = init_array(1, 1);
	b[0][0] = v;
	add(c, a, ar, ac, b, 1, 1);
	free_array(b);
}

void add(double ** c, double v, double ** b, int br, int bc) {
	double ** a = init_array(1, 1);
	a[0][0] = v;
	add(c, a, 1, 1, b, br, bc);
	free_array(a);
}

void sub(double ** c, double ** a, int ar, int ac, double v) {
	double ** b = init_array(1, 1);
	b[0][0] = v;
	sub(c, a, ar, ac, b, 1, 1);
	free_array(b);
}

void sub(double ** c, double v, double ** b, int br, int bc) {
	double ** a = init_array(1, 1);
	a[0][0] = v;
	sub(c, a, 1, 1, b, br, bc);
	free_array(a);
}

void mul(double ** c, double ** a, int ar, int ac, double v) {
	double ** b = init_array(1, 1);
	b[0][0] = v;
	mul(c, a, ar, ac, b, 1, 1);
	free_array(b);
}

void mul(double ** c, double v, double ** b, int br, int bc) {
	double ** a = init_array(1, 1);
	a[0][0] = v;
	mul(c, a, 1, 1, b, br, bc);
	free_array(a);
}

void div(double ** c, double ** a, int ar, int ac, double v) {
	double ** b = init_array(1, 1);
	b[0][0] = v;
	div(c, a, ar, ac, b, 1, 1);
	free_array(b);
}

void div(double ** c, double v, double ** b, int br, int bc) {
	double ** a = init_array(1, 1);
	a[0][0] = v;
	div(c, a, 1, 1, b, br, bc);
	free_array(a);
}

void pow(double ** c, double ** a, int ar, int ac, double v) {
	double ** b = init_array(1, 1);
	b[0][0] = v;
	pow(c, a, ar, ac, b, 1, 1);
	free_array(b);
}

void pow(double ** c, double v, double ** b, int br, int bc) {
	double ** a = init_array(1, 1);
	a[0][0] = v;
	pow(c, a, 1, 1, b, br, bc);
	free_array(a);
}

void save(double ** d, int r, int c, std :: ofstream& f) {
	for(int i = 0; i < r; i++) {
		for(int j = 0; j < c; j++) {
			f << d[i][j] << " ";
		}
	}
}

void load(double ** d, int r, int c, std :: ifstream& f) {
	for(int i = 0; i < r; i++) {
		for(int j = 0; j < c; j++) {
			f >> d[i][j];
		}
	}
}


void print(double ** a, int ar, int ac) {
	if (pid == 0) {
		for(int i = 0; i < ar; i++) {
			for(int j = 0; j < ac; j++) printf("%3.4f\t", a[i][j]);
			std :: cout << std :: endl;
		}
	}
}

# endif