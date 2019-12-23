# ifndef _tensor_
# define _tensor_

/*

LOP DINH NGHIA TENSOR 2 CHIEU
CAC THUOC TINH :
	int n, m : 				KICH THUOC CUA MA TRAN
	double ** d :			MANG 2 CHIEU KICH THUOC n * m, DU LIEU CUA MA TRAN
	double ** temp_grad : 	MANG 2 CHIEU KICH THUOC n * m, LUU DAO HAM TAM THOI CUA TENSOR
	double ** grad :		MANG 2 CHIEU KICH THUOC n * m, DAO HAM CUA TENSOR
	bool isvar : 			TENSOR CO PHAI LA VARIABLE HAY KHONG, CO DUOC PHEP CAP NHAT TRONG QUA TRINH TRAIN HAY KHONG
	Tensor * left : 		CON TRAI CUA TENSOR, NEU KHONG CO THI BANG Null
	Tensor * left : 		CON PHAI CUA TENSOR, NEU KHONG CO THI BANG Null
	string op :				OPERATOR CUA TENSOR
	double drop : 			DROPOUT
	double ** mask :		SU DUNG TRONG QUA TRINH TINH TOAN DROPOUT

CAC HAM :
	random_init() :			KHOI TAO GIA TRI NGAU NHIEN CHO DU LIEU CUA TENSOR
	print() :				IN TENSOR THEO DINH DANG
	run() :					TINH TOAN GIA TRI CUA TENSOR
	backward() :			TINH TOAN LOI CUA TOAN BO TENSOR GRAPH
	zero_grad() :			XOA TOAN BO GRADIENT CUA GRAPH
	void distribute():		PHAN PHOI GIA TRI CUA TENSOR CHO TUNG PROCESS
	void centralize():		TONG HOP GIA TRI CUA TENSOR O TAT CA CAC PROCESS
	void uniform():			DONG NHAT GIA TRI CUA CAC TENSOR (LAY TRUNG BINH)

CAC PHEP TOAN :
	+ :		PHEP CONG GIUA 2 TENSOR, GIUA TENSOR VOI MOT SO THUC, VA MOT SO THUC VOI TENSOR
	- :		PHEP TRU GIUA 2 TENSOR, GIUA TENSOR VOI MOT SO THUC, VA MOT SO THUC VOI TENSOR
	* :		PHEP NHAN GIUA 2 TENSOR, GIUA TENSOR VOI MOT SO THUC, VA MOT SO THUC VOI TENSOR
	/ :		PHEP CHIA GIUA 2 TENSOR, GIUA TENSOR VOI MOT SO THUC, VA MOT SO THUC VOI TENSOR
	^ :		PHEP MU GIUA 2 TENSOR, GIUA TENSOR VOI MOT SO THUC, VA MOT SO THUC VOI TENSOR
	% : 	PHEP NHAN MA TRAN GIUA 2 TENSOR
	log :	PHEP TOAN MOT NGOI TRA VE LOGARIT CO SO e CUA TENSOR
	exp :	PHEP TOAN MOT NGOI TRA VE EXP CUA TENSOR
	mean :	PHEP LAY GIA TRI TRUNG BINH
	sum :	PHEP LAY TONG
	abs :	PHEP LAY GIA TRI TUYET DOI

..................

*/


# include <bits/stdc++.h>
# include "tensor_operator.cpp"
# include <mpi.h>

# define RANDMAX 10000000000

std::random_device dev;
std::mt19937 rng(dev());
std::uniform_int_distribution<std::mt19937::result_type> dist(0, RANDMAX);

double random_() {
	return (double) dist(rng) / RANDMAX;
}

int max(int a, int b) {
	return a > b ? a : b;
}

class Tensor {
	public:
		Tensor();
		Tensor(double v);
		Tensor(int a, int b);
		Tensor(double ** dt, int a, int b);

		static int count;
		int n, m;
		double ** d;
		double ** temp_grad;
		double ** grad;
		double ** momentum;
		double ** momentum2;
		bool isvar;
		Tensor * left;
		Tensor * righ;
		Tensor * par;
		std :: string op;
		double drop;
		double ** mask;
		double ** left_mask;
		double ** righ_mask;
		int nn;
		bool visited;

		void random_init();
		void random_init(double x, double y);
		void from_array(double ** a, int ar, int ac);
		void is_var();
		void isnt_var();
		void dropout(double dr);

		Tensor& operator + (Tensor &other);
		Tensor& operator - (Tensor &other);
		Tensor& operator * (Tensor &other);
		Tensor& operator / (Tensor &other);
		Tensor& operator -();	
		Tensor& operator !();
		Tensor& operator % (Tensor &other);
		Tensor& operator + (double x);
		Tensor& operator - (double x);
		Tensor& operator * (double x);
		Tensor& operator / (double x);
		Tensor& operator ^ (Tensor &other);
		Tensor& operator ^ (double v);

		void distribute();
		void combine();

		void uniform();
		void _uniform();

		void build();
		void _build();

		void run();
		void _run();

		void _backward();
		void backward();

		void zero_grad();
		void reset_graph();

		void gd_step(double lr);
		void momentum_step(double lr, double gamma);
		void adagrad_step(double lr);
		void adadelta_step(double lr, double b1);
		void rmsprop_step(double lr, double b2);
		void adam_step(double lr, double b1, double b2, int t);

		void _save_graph(std :: ofstream& f);
		void save_graph(std::string filename);
		void _load_graph(std :: ifstream& f);
		void load_graph(std::string filename);
};

int Tensor :: count = -1000;
Tensor graph[1000];

Tensor :: Tensor() {
	count++; n = 0; m = 0;
	left = NULL; righ = NULL; par = NULL;
	isvar = 0; drop = 0; op = "0x";

	d = init_array(1, 0);
	grad = init_array(1, 0);
	temp_grad = init_array(1, 0);
	momentum = init_array(1, 0);
	momentum2 = init_array(1, 0);
	mask = init_array(1, 0);
	left_mask = init_array(1, 0);
	righ_mask = init_array(1, 0);
}

Tensor :: Tensor(double v){
	count++; n = 1, m = 1;
	left = NULL; righ = NULL; par = NULL;
	isvar = 0; drop = 0; op = "0x";

	d = init_array(1, 1);
	grad = init_array(1, 1);
	temp_grad = init_array(1, 1);
	momentum = init_array(1, 0);
	momentum2 = init_array(1, 0);
	mask = init_array(1, 0);
	left_mask = init_array(1, 0);
	righ_mask = init_array(1, 0);

	d[0][0] = v;
}

Tensor :: Tensor(int a, int b){
	if (a == 0 or b == 0) {
		count++; n = a; m = b;
		left = NULL; righ = NULL; par = NULL;
		isvar = 0; drop = 0; op = "0x";

		d = init_array(1, 0);
		grad = init_array(1, 0);
		temp_grad = init_array(1, 0);
		momentum = init_array(1, 0);
		momentum2 = init_array(1, 0);
		mask = init_array(1, 0);
		left_mask = init_array(1, 0);
		righ_mask = init_array(1, 0);
	}

	else {
		count++; n = a, m = b;
		left = NULL; righ = NULL; par = NULL;
		isvar = 0; drop = 0; op = "0x";

		d = init_array(n, m);
		grad = init_array(n, m);
		temp_grad = init_array(n, m);
		momentum = init_array(1, 0);
		momentum2 = init_array(1, 0);
		mask = init_array(1, 0);
		left_mask = init_array(1, 0);
		righ_mask = init_array(1, 0);
	}
}


Tensor :: Tensor(double ** dt, int a, int b) {
	count++; n = a, m = b;
	left = NULL; righ = NULL; par = NULL;
	isvar = 0; drop = 0; op = "0x";
	
	d = init_array(n, m);
	grad = init_array(n, m);
	temp_grad = init_array(n, m);
	momentum = init_array(1, 0);
	momentum2 = init_array(1, 0);
	mask = init_array(1, 0);
	left_mask = init_array(1, 0);
	righ_mask = init_array(1, 0);

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			d[i][j] = dt[i][j];
		}
	}
}

void Tensor :: dropout(double dr) {
	drop = dr;
}

void Tensor :: is_var() {
	isvar = true;
	free_array(momentum);	momentum = init_array(n, m);
	free_array(momentum2);	momentum2 = init_array(n, m);	
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			momentum[i][j] = 0;
			momentum2[i][j] = 0;
		}
	}
}

void Tensor :: isnt_var() {
	isvar = false;
	free_array(momentum);	momentum = init_array(1, 0);
	free_array(momentum2);	momentum2 = init_array(1, 0);	
}


void print(Tensor t) {
	std :: cout << "Tensor ("<< t.n << " " << t.m << ")\t Is Variable: "<< t.isvar << std :: endl;
	for(int i = 0; i < t.n; i++) {
		for(int j = 0; j < t.m; j++) printf("%3.4f\t", t.d[i][j]);
		std :: cout << std :: endl;
	}
}

void _print(Tensor t) {
	std :: cout << "Tensor ("<< t.n << " " << t.m << ")\t Is Variable: "<< t.isvar << std :: endl;
	for(int i = 0; i < t.n; i++) {
		for(int j = 0; j < t.m; j++) printf("%3.4f\t", t.grad[i][j]);
		std :: cout << std :: endl;
	}
}

/*	
	DINH NGHIA CAC PHEP TOAN TREN TENSOR
	CAC PHEP TOAN KHONG TINH TOAN NGAY GIA TRI CUA TENSOR, CHI TAO DO THI DONG TENSOR
*/

/* ***** PHEP CONG ****** */




Tensor& Tensor :: operator + (Tensor &other) {
	int x = count;
	graph[x] = Tensor();

	par = &(graph[x]);
	other.par = &(graph[x]);

	(graph[x]).left = this;
	(graph[x]).righ = &other;

	(graph[x]).op = "add";

	return graph[x];
}

Tensor& operator + (double v, Tensor &t) {
	int x = t.count;

	graph[x] = Tensor(v);
	graph[x + 1] = Tensor();

	(graph[x + 1]).righ = &t;
	(graph[x + 1]).left = &(graph[x]);

	t.par = &(graph[x + 1]);
	(graph[x]).par = &(graph[x + 1]);

	(graph[x + 1]).op = "add";

	return graph[x + 1];
}

Tensor& Tensor :: operator + (double v){
	int x = count;

	graph[x] = Tensor(v);
	graph[x + 1] = Tensor();

	(graph[x + 1]).left = this;
	(graph[x + 1]).righ = &(graph[x]);

	par = &(graph[x]);
	(graph[x]).par = &(graph[x + 1]);

	(graph[x + 1]).op = "add";

	return graph[x + 1];
}

/* ***** PHEP TRU ****** */

Tensor& Tensor :: operator - (Tensor &other) {
	int x = count;
	graph[x] = Tensor();

	par = &(graph[x]);
	other.par = &(graph[x]);

	(graph[x]).left = this;
	(graph[x]).righ = &other;

	(graph[x]).op = "sub";

	return graph[x];
}

Tensor& operator - (double v, Tensor &t) {
	int x = t.count;

	graph[x] = Tensor(v);
	graph[x + 1] = Tensor();

	(graph[x + 1]).righ = &t;
	(graph[x + 1]).left = &(graph[x]);

	t.par = &(graph[x + 1]);
	(graph[x]).par = &(graph[x + 1]);

	(graph[x + 1]).op = "sub";

	return graph[x + 1];
}

Tensor& Tensor :: operator - (double v){
	int x = count;

	graph[x] = Tensor(v);
	graph[x + 1] = Tensor();

	(graph[x + 1]).left = this;
	(graph[x + 1]).righ = &(graph[x]);

	par = &(graph[x]);
	(graph[x]).par = &(graph[x + 1]);

	(graph[x + 1]).op = "sub";
	
	return graph[x + 1];
}

/* ***** PHEP NHAN ****** */

Tensor& Tensor :: operator * (Tensor &other) {
	int x = count;
	graph[x] = Tensor();

	par = &(graph[x]);
	other.par = &(graph[x]);

	(graph[x]).left = this;
	(graph[x]).righ = &other;

	(graph[x]).op = "mul";

	return graph[x];
}

Tensor& operator * (double v, Tensor &t) {
	int x = t.count;

	graph[x] = Tensor(v);
	graph[x + 1] = Tensor();

	(graph[x + 1]).righ = &t;
	(graph[x + 1]).left = &(graph[x]);

	t.par = &(graph[x + 1]);
	(graph[x]).par = &(graph[x + 1]);

	(graph[x + 1]).op = "mul";

	return graph[x + 1];
}

Tensor& Tensor :: operator * (double v){
	int x = count;

	graph[x] = Tensor(v);
	graph[x + 1] = Tensor();

	(graph[x + 1]).left = this;
	(graph[x + 1]).righ = &(graph[x]);

	par = &(graph[x]);
	(graph[x]).par = &(graph[x + 1]);

	(graph[x + 1]).op = "mul";
	
	return graph[x + 1];
}

/* ***** PHEP CHIA ****** */

Tensor& Tensor :: operator / (Tensor &other) {
	int x = count;
	graph[x] = Tensor();

	par = &(graph[x]);
	other.par = &(graph[x]);

	(graph[x]).left = this;
	(graph[x]).righ = &other;

	(graph[x]).op = "div";

	return graph[x];
}

Tensor& operator / (double v, Tensor &t) {
	int x = t.count;

	graph[x] = Tensor(v);
	graph[x + 1] = Tensor();

	(graph[x + 1]).righ = &t;
	(graph[x + 1]).left = &(graph[x]);

	t.par = &(graph[x + 1]);
	(graph[x]).par = &(graph[x + 1]);

	(graph[x + 1]).op = "div";

	return graph[x + 1];
}

Tensor& Tensor :: operator / (double v){
	int x = count;

	graph[x] = Tensor(v);
	graph[x + 1] = Tensor();

	(graph[x + 1]).left = this;
	(graph[x + 1]).righ = &(graph[x]);

	par = &(graph[x]);
	(graph[x]).par = &(graph[x + 1]);

	(graph[x + 1]).op = "div";
	
	return graph[x + 1];
}

/* ***** PHEP LUY THUA ****** */

Tensor& Tensor :: operator ^ (Tensor &other) {
	int x = count;
	graph[x] = Tensor();

	par = &(graph[x]);
	other.par = &(graph[x]);

	(graph[x]).left = this;
	(graph[x]).righ = &other;

	(graph[x]).op = "pow";

	return graph[x];
}

Tensor& operator ^ (double v, Tensor &t) {
	int x = t.count;

	graph[x] = Tensor(v);
	graph[x + 1] = Tensor();

	(graph[x + 1]).righ = &t;
	(graph[x + 1]).left = &(graph[x]);

	t.par = &(graph[x + 1]);
	(graph[x]).par = &(graph[x + 1]);

	(graph[x + 1]).op = "pow";

	return graph[x + 1];
}

Tensor& Tensor :: operator ^ (double v){
	int x = count;

	graph[x] = Tensor(v);
	graph[x + 1] = Tensor();

	(graph[x + 1]).left = this;
	(graph[x + 1]).righ = &(graph[x]);

	par = &(graph[x]);
	(graph[x]).par = &(graph[x + 1]);

	(graph[x + 1]).op = "pow";
	
	return graph[x + 1];
}

/* ***** PHEP NHAN MA TRAN ****** */

Tensor& Tensor :: operator % (Tensor &other) {
	int x = count;

	graph[x] = Tensor();

	par = &(graph[x]);
	other.par = &(graph[x]);

	(graph[x]).left = this;
	(graph[x]).righ = &other;

	(graph[x]).op = "matmul";

	return graph[x];
}

/* ***** PHEP LAY NGHICH DAO ****** */

Tensor& Tensor :: operator - () {
	int x = count;
	graph[x] = Tensor();

	(graph[x]).left = this;
	par = &(graph[x]);

	(graph[x]).op = "negative";

	return graph[x];
}

/* ***** MA TRAN CHUYEN VI ****** */

Tensor& Tensor :: operator !() {
	int x = count;
	graph[x] = Tensor();

	par = &(graph[x]);
	(graph[x]).left = this;

	(graph[x]).op = "transpose";

	return (graph[x]);
}

/* ***** LOGARIT ****** */

Tensor& log(Tensor &t) {
	int x = t.count;
	graph[x] = Tensor();

	(graph[x]).left = &t;
	t.par = &(graph[x]);

	(graph[x]).op = "log";

	return graph[x];
}

/* ***** EXP ****** */

Tensor& exp(Tensor &t) {
	int x = t.count;
	graph[x] = Tensor();

	(graph[x]).left = &t;
	t.par = &(graph[x]);

	(graph[x]).op = "exp";

	return graph[x];
}

Tensor& abs(Tensor &t) {
	int x = t.count;
	graph[x] = Tensor();

	(graph[x]).left = &t;
	t.par = &(graph[x]);

	(graph[x]).op = "abs";

	return graph[x];
}


/* 
	***** PHEP LAY TRUNG BINH ****** 

	HAM MEAN TRA VE GIA TRI TRUNG BINH TREN TOAN MA TRAN 	(SIZE : 1, 1)
	HAM RMEAN TRA VE GIA TRI TRUNG BINH CUA TUNG COT 		(SIZE : 1, m)
	HAM CMEAN TRA VE GIA TRI TRUNG BINH CUA TUNG HANG		(SIZE : n, 1)

*/

Tensor& mean(Tensor &t) {
	int x = t.count;
	graph[x] = Tensor();

	(graph[x]).left = &t;
	t.par = &(graph[x]);

	(graph[x]).op = "mean";

	return graph[x];
}

Tensor& rmean(Tensor &t) {
	int x = t.count;
	graph[x] = Tensor();

	(graph[x]).left = &t;
	t.par = &(graph[x]);

	(graph[x]).op = "rmean";

	return graph[x];
}

Tensor& cmean(Tensor &t) {
	int x = t.count;
	graph[x] = Tensor();

	(graph[x]).left = &t;
	t.par = &(graph[x]);

	(graph[x]).op = "cmean";

	return graph[x];
}

/* ***** PHEP LAY TONG ****** */

Tensor& sum(Tensor &t) {
	int x = t.count;
	graph[x] = Tensor();

	(graph[x]).left = &t;
	t.par = &(graph[x]);

	(graph[x]).op = "sum";

	return graph[x];
}

Tensor& rsum(Tensor &t) {
	int x = t.count;
	graph[x] = Tensor();

	(graph[x]).left = &t;
	t.par = &(graph[x]);

	(graph[x]).op = "rsum";

	return graph[x];
}

Tensor& csum(Tensor &t) {
	int x = t.count;
	graph[x] = Tensor();

	(graph[x]).left = &t;
	t.par = &(graph[x]);

	(graph[x]).op = "csum";

	return graph[x];
}

Tensor& max(Tensor &a, Tensor &b) {
	int x = a.count;
	graph[x] = Tensor();

	(graph[x]).left = &a;
	(graph[x]).righ = &b;

	a.par = &(graph[x]);
	b.par = &(graph[x]);

	(graph[x]).op = "max";

	return graph[x];
}

Tensor& max(Tensor &a, double v) {
	int x = a.count;

	graph[x] = Tensor(v);
	graph[x + 1] = Tensor();

	(graph[x + 1]).left = &a;
	(graph[x + 1]).righ = &(graph[x]);

	a.par = &(graph[x + 1]);
	(graph[x]).par = &(graph[x + 1]);

	(graph[x + 1]).op = "max";

	return graph[x + 1];
}

Tensor& max(double v, Tensor &a) {
	int x = a.count;

	graph[x] = Tensor(v);
	graph[x + 1] = Tensor();

	(graph[x + 1]).righ = &a;
	(graph[x + 1]).left = &(graph[x]);

	a.par = &(graph[x + 1]);
	(graph[x]).par = &(graph[x + 1]);

	(graph[x + 1]).op = "max";

	return graph[x + 1];
}

Tensor& min(Tensor &a, Tensor &b) {
	int x = a.count;
	graph[x] = Tensor();

	(graph[x]).left = &a;
	(graph[x]).righ = &b;

	a.par = &(graph[x]);
	b.par = &(graph[x]);

	(graph[x]).op = "min";

	return graph[x];
}

Tensor& min(Tensor &a, double v) {
	int x = a.count;

	graph[x] = Tensor(v);
	graph[x + 1] = Tensor();

	(graph[x + 1]).left = &a;
	(graph[x + 1]).righ = &(graph[x]);

	a.par = &(graph[x + 1]);
	(graph[x]).par = &(graph[x + 1]);

	(graph[x + 1]).op = "min";

	return graph[x + 1];
}

Tensor& min(double v, Tensor &a) {
	int x = a.count;

	graph[x] = Tensor(v);
	graph[x + 1] = Tensor();

	(graph[x + 1]).righ = &a;
	(graph[x + 1]).left = &(graph[x]);

	a.par = &(graph[x + 1]);
	(graph[x]).par = &(graph[x + 1]);

	(graph[x + 1]).op = "min";
	
	return graph[x + 1];
}



/* ***************************** HAM KHOI TAO GIA TRI CUA TENSOR ********************* */

void Tensor :: from_array(double ** a, int ar, int ac) {
	n = ar; m = ac;
	free_array(d);
	d = init_array(ar, ac);
	for(int i = 0; i < ar; i++) {
		for(int j = 0; j < ac; j++) {
			d[i][j] = a[i][j];
		}
	}
}


void Tensor :: random_init() {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			d[i][j] = random_() - 0.5;
		}
	}
}

void Tensor :: random_init(double x, double y) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			d[i][j] = x + random_() * (y - x);
		}
	}
}
/* ******************************* TINH TOAN GIA TRI CUA TENSOR *************************
SU DUNG PHUONG PHAP DUYET DO THI
	- TINH GIA TRI CUA TENSOR LEFT, GIA TRI CUA TENSOR RIGHT
	- GIA TRI CUA TENSOR ROOT = GIA TRI CUA TENSOR LEFT [OPERATOR] GIA TRI CUA TENSOR RIGHT
	- DE QUY CHO DEN NUT LA

*/


void Tensor :: _build() {
	if (visited) return;
	if (op == "0x") {
		visited = true;
		return;
	}

	if (left != NULL) {
		left->_build();
	}
	if (righ != NULL) {
		righ->_build();
	}

	// std::cout << op << std::endl;

	if (op == "add"	or op == "sub" or op == "mul" or op == "div" or op == "pow") {

		if (	(left->n == righ->n and left->m == righ->m) or 
				((left->n == 1 or righ->n == 1) and left->m == righ->m) or
				(left->n == righ->n and (left->m == 1 or righ->m == 1)) or
				(left->n == 1 and left->m == 1) or (righ->n == 1 and righ->m == 1)		) {
		
			n = std::max(left->n, righ->n);
			m = std::max(left->m, righ->m);
		}

		else {
			throw "BUILD ERROR";
		}
	}

	
	if (op == "matmul") {
		if (left->m == righ->n) {
			n = left->n; m = righ->m;
		}
		else {
			throw "BUILD ERROR";
		}

	}

	if (op == "negative" or op == "abs" or op == "log" or op == "exp") {
		n = left->n; m = left->m;
	}

	if (op == "transpose") {
		n = left->m; m = left->n;
	}

	if (op == "mean" or op == "sum") {
		n = 1; m = 1;
	}

	if(op == "rmean" or op == "rsum") {
		n = 1; m = left->m;
	}

	if (op == "cmean" or op == "csum") {
		n = left->n; m = 1;
	}

	if (op == "max" or op == "min") {
		
		if (	(left->n == righ->n and left->m ==  righ->m) or
				(left->n == 1 and left->m == 1) or
				(righ->n == 1 and righ->m == 1)					) {

			n = std::max(left->n, righ->n);
			m = std::max(left->m, righ->m);

			free_array(left_mask); left_mask = init_array(n, m);
			free_array(righ_mask); righ_mask = init_array(n, m);
		}

		else {
			throw "BUILD ERROR";
		}
	}

	free_array(d); 			d = init_array(n, m);
	free_array(grad); 		grad = init_array(n, m);
	free_array(temp_grad); 	temp_grad = init_array(n, m);

	if (drop > 0){
		free_array(mask);
		mask = init_array(n, m);
	}

	visited = true;
}

void Tensor :: build() {
	reset_graph();
	_build();
}



void Tensor :: _run() {

	if (visited) return;
	if (left != NULL) {
		left->_run();
	}
	if (righ != NULL) {
		righ->_run();
	}


	if (op == "add") {
		add(d, left->d, left->n, left->m, righ->d, righ->n, righ->m);
	}

	if (op == "sub") {
		sub(d, left->d, left->n, left->m, righ->d, righ->n, righ->m);
	}

	if (op == "mul") {
		mul(d, left->d, left->n, left->m, righ->d, righ->n, righ->m);
	}

	if (op == "div") {
		div(d, left->d, left->n, left->m, righ->d, righ->n, righ->m);
	}

	if (op == "pow") {
		pow(d, left->d, left->n, left->m, righ->d, righ->n, righ->m);
	}

	if(op == "matmul") {
		matmul(d, left->d, left->n, left->m, righ->d, righ->n, righ->m);
	}

	if (op == "negative") {
		negative(d, left->d, left->n, left->m);
	}

	if (op == "abs") {
		abs(d, left->d, left->n, left->m);
	}

	if (op == "transpose") {
		transpose(d, left->d, left->n, left->m);
	}

	if (op == "log") {
		log(d, left->d, left->n, left->m);
	}

	if (op == "exp") {
		exp(d, left->d, left->n, left->m);
	}

	if (op == "mean") {
		mean(d, left->d, left->n, left->m);
	}

	if(op == "rmean") {
		rmean(d, left->d, left->n, left->m);
	}

	if (op == "cmean") {
		cmean(d, left->d, left->n, left->m);
	}

	if (op == "sum") {
		sum(d, left->d, left->n, left->m);
	}

	if (op == "rsum") {
		rsum(d, left->d, left->n, left->m);
	}

	if (op == "csum") {
		csum(d, left->d, left->n, left->m);
	}

	if (op == "max") {
		maximum(d, left->d, left->n, left->m, righ->d, righ->n,
					righ->m, left_mask, righ_mask);
	}

	if (op == "min") {
		minimum(d, left->d, left->n, left->m, righ->d, righ->n,
				righ->m, left_mask, righ_mask);
	}

	if (drop > 0){
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {
				if (random_() < drop) {
					mask[i][j] = 0;
				}
				else {
					mask[i][j] = 1 / (1 - drop);
				}
			}
		}
		
		mul(d, d, n, m, mask, n, m);
	}
	visited = true;
}

void Tensor :: run() {
	reset_graph();
	_run();
}

/* SU DUNG BACK PROPAGATION DE TINH TOAN ERROR CUA TAT CA CAC TENSOR TRONG GRAPH

*/

void Tensor :: _backward() {

	if (drop > 0) {
		mul(temp_grad, temp_grad, n, m, mask, n, m);
		mul(grad, grad, n, m, mask, n, m);
	}

	/******** MEAN, SUM ***********/
	// if (pid == 0) std :: cout << op << std :: endl;
	if (op == "mean") {
		fill(left->temp_grad, left->n, left->m, temp_grad[0][0] / (left->n * left->m));
	}

	if (op == "rmean") { //
		for(int i = 0; i < left->n; i++) {
			for(int j = 0; j < left->m; j++) {
				left->temp_grad[i][j] = temp_grad[0][j] / left->n;
			}
		}
	}

	if (op == "cmean") {
		for(int i = 0; i < left->n; i++) {
			for(int j = 0; j < left->m; j++) {
				left->temp_grad[i][j] = temp_grad[i][0] / left->m;
			}
		}
	}

	if (op == "sum") {
		fill(left->temp_grad, left->n, left->m, temp_grad[0][0]);
	}

	if (op == "rsum") { //
		for(int i = 0; i < left->n; i++) {
			for(int j = 0; j < left->m; j++) {
				left->temp_grad[i][j] = temp_grad[0][j];
			}
		}
	}

	if (op == "csum") {
		for(int i = 0; i < left->n; i++) {
			for(int j = 0; j < left->m; j++) {
				left->temp_grad[i][j] = temp_grad[i][0];
			}
		}
	}

	/********* PHEP CONG ************/

	if (op == "add") {
		if (left->n == n and left->m == m) {
			copy(left->temp_grad, temp_grad, n, m);
		}
		else if (left->n != n and left->m == m) {
			rsum(left->temp_grad, temp_grad, n, m);
		}
		else if (left->m != m and left->n == n){
			csum(left->temp_grad, temp_grad, n, m);
		}
		else if(left->n == 1 and left->m == 1) {
			sum(left->temp_grad, temp_grad, n, m);
		}
		
		if (righ->n == n and righ->m == m) {
			copy(righ->temp_grad, temp_grad, n, m);
		}
		else if (righ->n != n and righ->m == m) {
			rsum(righ->temp_grad, temp_grad, n, m);
		}
		else if (righ->m != m and righ->n == n){
			csum(righ->temp_grad, temp_grad, n, m);
		}

		else if(righ->n == 1 and righ->m == 1) {
			sum(righ->temp_grad, temp_grad, n, m);
		}
	}

	/********* PHEP TRU ************/

	if (op == "sub") {
		double ** t1 = init_array(n, m);
		negative(t1, temp_grad, n, m);

		if (left->n == n and left->m == m) {
			copy(left->temp_grad, temp_grad, n, m);
		}
		else if (left->n != n and left->m == m) {
			rsum(left->temp_grad, temp_grad, n, m);
		}

		else if (left->m != m and left->n == n){
			csum(left->temp_grad, temp_grad, n, m);
		}
		else if(left->n == 1 and left->m == 1) {
			sum(left->temp_grad, temp_grad, n, m);
		}
		
		if (righ->n == n and righ->m == m) {
			copy(righ->temp_grad, t1, n, m);
		}
		else if (righ->n != n and righ->m == m) {
			rsum(righ->temp_grad, t1, n, m);
		}

		else if (righ->m != m and righ->n == n){
			csum(righ->temp_grad, t1, n, m);
		}
		else if(righ->n == 1 and righ->m == 1) {
			sum(righ->temp_grad, t1, n, m);
		}

		free_array(t1);
	}

	/********* PHEP NHAN ************/

	if (op == "mul") {
		double ** t1 = init_array(n, m);
		double ** t2 = init_array(n ,m);

		mul(t1, temp_grad, n, m, righ->d, righ->n, righ->m);
		mul(t2, temp_grad, n, m, left->d, left->n, left->m);


		if (left->n == n and left->m == m) {
			copy(left->temp_grad, t1, n, m);
		}
		else if (left->n != n and left->m == m) {
			rsum(left->temp_grad, t1, n, m);
		}
		else if (left->m != m and left->n == n){
			csum(left->temp_grad, t1, n, m);
		}
		else if(left->n == 1 and left->m == 1) {
			sum(left->temp_grad, t1, n, m);
		}
		
		if (righ->n == n and righ->m == m) {
			copy(righ->temp_grad, t2, n, m);
		}
		else if (righ->n != n and righ->m == m) {
			rsum(righ->temp_grad, t2, n, m);
		}
		else if (righ->m != m and righ->n == n){
			csum(righ->temp_grad, t2, n, m);
		}
		else if(righ->n == 1 and righ->m == 1) {
			sum(righ->temp_grad, t2, n, m);
		}
		free_array(t1); free_array(t2);
	}

	/* PHEP CHIA */

	if (op == "div") {

		double ** t0 = init_array(n, m);
		div(t0, temp_grad, n, m, righ->d, righ->n, righ->m);
		
		double ** t1 = init_array(righ->n, righ->m);
		double ** t2 = init_array(left->n, left->m);
		double ** t3 = init_array(n, m);
		double ** t4 = init_array(n, m);
		
		mul(t1, righ->d, righ->n, righ->m, righ->d, righ->n, righ->m);
		div(t2, left->d, left->n, left->m, t1, righ->n, righ->m);
		mul(t3, temp_grad, n, m, t2, left->n, left->m);
		negative(t4, t3, n, m);

		if (left->n == n and left->m == m) {
			copy(left->temp_grad, t0, n, m);
		}
		else if (left->n != n and left->m == m) {
			rsum(left->temp_grad, t0, n, m);
		}
		else if (left->m != m and left->n == n){
			csum(left->temp_grad, t0, n, m);
		}
		else if(left->n == 1 and left->m == 1) {
			sum(left->temp_grad, t0, n, m);
		}

		if (righ->n == n and righ->m == m) {
			copy(righ->temp_grad, t4, n, m);
		}
		else if (righ->n != n and righ->m == m) {
			rsum(righ->temp_grad, t4, n, m);
		}
		else if (righ->m != m and righ->n == n){
			csum(righ->temp_grad, t4, n, m);
		}
		else if(righ->n == 1 and righ->m == 1) {
			sum(righ->temp_grad, t4, n, m);
		}

		free_array(t0); free_array(t1); free_array(t2); free_array(t3); free_array(t4);

	}

	/* PHEP LUY THUA */

	if (op == "pow") {
		
		double ** t1 = init_array(righ->n, righ->m);
		double ** t2 = init_array(n, m);
		double ** t3 = init_array(n, m);
		double ** t0 = init_array(n, m);

		sub(t1, righ->d, righ->n, righ->m, 1);
		pow(t2, left->d, left->n, left->m, t1, righ->n, righ->m);
		mul(t3, t2, n, m, righ->d, righ->n, righ->m);
		mul(t0, t3, n, m, temp_grad, n, m);

		double ** t4 = init_array(left->n, left->m);
		double ** t5 = init_array(n, m);
		double ** t6 = init_array(n, m);

		exp(t4, left->d, left->n, left->m);
		mul(t5, t4, left->n, left->m, d, n, m);
		mul(t6, t5, n, m, temp_grad, n, m);

		if (left->n == n and left->m == m) {
			copy(left->temp_grad, t0, n, m);
		}
		else if (left->n != n and left->m == m) {
			rsum(left->temp_grad, t0, n, m);
		}
		else if (left->m != m and left->n == n){
			csum(left->temp_grad, t0, n, m);
		}
		else if(left->n == 1 and left->m == 1) {
			sum(left->temp_grad, t0, n, m);
		}
		
		if (righ->n == n and righ->m == m) {
			copy(righ->temp_grad, t6, n, m);
		}
		else if (righ->n != n and righ->m == m) {
			rsum(righ->temp_grad, t6, n, m);
		}
		else if (righ->m != m and righ->n == n){
			csum(righ->temp_grad, t6, n, m);
		}
		else if(righ->n == 1 and righ->m == 1) {
			sum(righ->temp_grad, t6, n, m);
		}

		free_array(t0); free_array(t1); free_array(t2); free_array(t3);
		free_array(t4); free_array(t5); free_array(t6);
	}

	/********* PHEP NHAN MA TRAN ************/

	if (op == "matmul") {
		double ** t1 = init_array(left->m, left->n);
		double ** t2 = init_array(righ->m, righ->n);

		transpose(t1, left->d, left->n, left->m);
		transpose(t2, righ->d, righ->n, righ->m);

		matmul(righ->temp_grad, t1, left->m, left->n, temp_grad, n, m);
		matmul(left->temp_grad, temp_grad, n, m, t2, righ->m, righ->n);

		free_array(t1); free_array(t2);

	}

	/* ******************************************* */

	if (op == "transpose") {
		transpose(left->temp_grad, temp_grad, n, m);
	}

	if (op == "negative") {
		negative(left->temp_grad, temp_grad, n, m);
	}

	if (op == "abs") {
		double ** t1 = init_array(n, m);
		sign(t1, left->d, n, m);
		mul(left->temp_grad, temp_grad, n, m, t1, n, m);
		free_array(t1);
	}

	if (op == "log") {
		div(left->temp_grad, temp_grad, n, m, left->d, n, m);
	}

	if (op == "exp") {
		mul(left->temp_grad, temp_grad, n, m, d, n, m);
	}

	if (op == "max" or op == "min") {
		
		double ** t1 = init_array(n, m);
		double ** t2 = init_array(n, m);
		mul(t1, temp_grad, n, m, left_mask, n, m);
		mul(t2, temp_grad, n, m, righ_mask, n, m);

		if (left->n == n and left->m == m) {
			copy(left->temp_grad, t1, n, m);
		}
		else if (left->n != n and left->m == m) {
			rsum(left->temp_grad, t1, n, m);
		}
		else if (left->m != m and left->n == n){
			csum(left->temp_grad, t1, n, m);
		}
		else if(left->n == 1 and left->m == 1) {
			sum(left->temp_grad, t1, n, m);
		}
		
		if (righ->n == n and righ->m == m) {
			copy(righ->temp_grad, t2, n, m);
		}
		else if (righ->n != n and righ->m == m) {
			rsum(righ->temp_grad, t2, n, m);
		}
		else if (righ->m != m and righ->n == n){
			csum(righ->temp_grad, t2, n, m);
		}
		else if(righ->n == 1 and righ->m == 1) {
			sum(righ->temp_grad, t2, n, m);
		}
		free_array(t1); free_array(t2);
	}

	if(left != NULL) add(left->grad, left->grad, left->n, left->m, left->temp_grad, left->n, left->m);
	if(righ != NULL) add(righ->grad, righ->grad, righ->n, righ->m, righ->temp_grad, righ->n, righ->m);	

	if(left != NULL) left->_backward();
	if(righ != NULL) righ->_backward();
}

void Tensor :: backward() {
	fill(grad, n, m, 1);
	fill(temp_grad, n, m, 1);
	_backward();
}

void Tensor :: zero_grad() {
	memset(&grad[0][0], 0, n * m * sizeof(double));
	if(left != NULL) {
		left->zero_grad();
	}
	if (righ != NULL) {
		righ->zero_grad();
	}
}

void Tensor :: reset_graph() {
	visited = false;
	if(left != NULL) {
		left->reset_graph();
	}
	if (righ != NULL) {
		righ->reset_graph();
	}
}

void Tensor :: gd_step(double lr) {
	if (isvar) {
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {
				d[i][j] -= lr * grad[i][j];
			}
		}
	}
	if (left != NULL) {
		left->gd_step(lr);
	}
	if (righ != NULL) {
		righ->gd_step(lr);
	}
}

void Tensor :: momentum_step(double lr, double gamma) {
	if (isvar) {
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {
				momentum[i][j] = momentum[i][j] * gamma + (1 - gamma) * grad[i][j];
				d[i][j] -= lr * momentum[i][j];
			}
		}
	}
	if (left != NULL) {
		left->momentum_step(lr, gamma);
	}
	if (righ != NULL) {
		righ->momentum_step(lr, gamma);
	}
}

void Tensor :: adagrad_step(double lr) {
	if (isvar) {
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {
				momentum[i][j] = momentum[i][j] + grad[i][j] * grad[i][j];
				d[i][j] -= lr * grad[i][j] / sqrt(momentum[i][j] + 1e-6);
			}
		}
	}
	if (left != NULL) {
		left->adagrad_step(lr);
	}
	if (righ != NULL) {
		righ->adagrad_step(lr);
	}
}

void Tensor :: adadelta_step(double lr, double b1) {
	if (isvar) {
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {
				momentum[i][j] = b1 * momentum[i][j] + (1 - b1) * grad[i][j] * grad[i][j];
				double g = sqrt((momentum2[i][j] + 1e-6) / (momentum[i][j] + 1e-6)) * grad[i][j];
				d[i][j] -= lr * g;
				momentum2[i][j] = b1 * momentum2[i][j] + (1 - b1) * d[i][j] * d[i][j];
			}
		}
	}
	if (left != NULL) {
		left->adadelta_step(lr, b1);
	}
	if (righ != NULL) {
		righ->adadelta_step(lr, b1);
	}
}

void Tensor :: rmsprop_step(double lr, double b1) {
	if (isvar) {
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {
				momentum[i][j] = b1 * momentum[i][j] + (1 - b1) * grad[i][j] * grad[i][j];
				d[i][j] -= lr * grad[i][j] / sqrt(momentum[i][j] + 1e-6);
			}
		}
	}
	if (left != NULL) {
		left->rmsprop_step(lr, b1);
	}
	if (righ != NULL) {
		righ->rmsprop_step(lr, b1);
	}
}

void Tensor :: adam_step(double lr, double b1, double b2, int t) {
	if (isvar) {
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {
				momentum[i][j] = b1 * momentum[i][j] + (1 - b1) * grad[i][j];
				momentum2[i][j] = b2 * momentum2[i][j] + (1 - b2) * grad[i][j] * grad[i][j];
				double vt = momentum[i][j] / (1 - pow(b1, t));
				double st = momentum2[i][j] / (1 - pow(b2, t));
				d[i][j] -= lr * vt / (sqrt(st) + 1e-6);

			}
		}
	}
	if (left != NULL) {
		left->adam_step(lr, b1, b2, t);
	}
	if (righ != NULL) {
		righ->adam_step(lr, b1, b2, t);
	}
}

void Tensor :: distribute() {
	double ** data = init_array(n, m);
	copy(data, d, n, m);
	int rr = n / np;
	int ri = pid < n % np ? rr + 1 : rr;
	int ofs = pid < n % np ? ri * pid : (ri + 1) * pid - (pid - (n % np));
	nn = n;
	free_array(d);
	n = ri;
	d = init_array(n, m);
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			d[i][j] = data[ofs + i][j];
		}
	}
	free_array(data);
}

void Tensor :: _uniform() {
	// std :: cout << pid << n << " " << m << std::endl;
	int num = np;
	int tag = 0;
	int des, src;
	while (num > 1) {
		tag++;
		int z = num / 2 + num % 2;
		if (pid >= z and pid < num) {
			des = pid - z;
			MPI_Send(&d[0][0], n * m, MPI_DOUBLE, des, tag, MPI_COMM_WORLD);
		}
		else if (pid < z) {
			src = pid + z;
			double ** t = init_array(n, m);
			MPI_Recv(&t[0][0], n * m, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);
			for(int i = 0; i < n; i++) {
				for(int j = 0; j < m; j++) {
					d[i][j] += t[i][j];
				}
			}
			free_array(t);
		}
		tag++;
		num = z;
	}

	if (pid == 0) {
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {
				d[i][j] /= np;
			}
		}
	}

	while (num < np) {
		if (pid < num) {
			des = pid + num;
			if (des < np) {
				MPI_Send(&d[0][0], n * m, MPI_DOUBLE, des, tag, MPI_COMM_WORLD);	
			}
		}
		else if(pid < 2 * num) {
			src = pid - num;
			MPI_Recv(&d[0][0], n * m, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);
		}
		num *= 2;
		tag++;
	}
}

void Tensor :: uniform() {
	if (isvar) _uniform();
	if(left != NULL) {
		left->uniform();
	}
	if (righ != NULL) {
		righ->uniform();
	}
}

void Tensor :: combine() {
	int tag = 10000;
	if (pid == 0) {
		double ** data = init_array(n * np, m);
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++) {
				data[i][j] = d[i][j];
			}
		}
		int ofs = n;
		int ri;
		for(int i = 1; i < np; i++) {
			MPI_Recv(&ri, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
			MPI_Recv(&data[ofs][0], ri * m, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
			ofs += ri;
		}
		free_array(d);
		if (op == "mean" or op == "rmean") {
			n = 1;
			d = init_array(1, m);
			rmean(d, data, ofs, m);
		}
		else if (op == "sum" or op == "rsum") {
			n = 1;
			d = init_array(1, m);
			rsum(d, data, ofs, m);
		}
		else {
			n = ofs;
			d = init_array(n, m);
			copy(d, data, n, m);
		}
		
		tag++;
		for(int i = 1; i < np; i++) {
			MPI_Send(&n, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
			MPI_Send(&d[0][0], n * m, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
		}
	}
	else {
		int ri = n;
		MPI_Send(&ri, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
		MPI_Send(&d[0][0], ri * m, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
		free_array(d);
		tag++;
		MPI_Recv(&n, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
		d = init_array(n, m);
		MPI_Recv(&d[0][0], n * m, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
	}	
}

void Tensor :: save_graph(std :: string filename) {
	reset_graph();
	std :: ofstream f;
	f.open(filename);
	_save_graph(f);
	f.close();
}

void Tensor :: _save_graph(std :: ofstream& f) {
	if (visited) return;
	if (isvar) save(d, n, m, f);
	if (left != NULL) left->_save_graph(f);
	if (righ != NULL) righ->_save_graph(f);
	visited = true;
}

void Tensor :: load_graph(std :: string filename) {
	reset_graph();
	std :: ifstream f;
	f.open(filename);
	_load_graph(f);
	f.close();
}

void Tensor :: _load_graph(std :: ifstream& f) {
	if (visited) return;
	if (isvar) load(d, n, m, f);
	if (left != NULL) left->_load_graph(f);
	if (righ != NULL) righ->_load_graph(f);
	visited = true;
}

# endif