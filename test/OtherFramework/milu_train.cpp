#include <bits/stdc++.h>
#include "../../milu.h"
#include "mpi.h"

using namespace std;

int main() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    Tensor x = Tensor(1000, 30); x.random_init();
    Tensor y = Tensor(1000, 10); y.random_init();

    Tensor h = relu(linear(x, 30, 50));
    Tensor z = linear(h, 50, 10);

    Tensor loss = MSELoss(y, z);

    x.distribute();
    y.distribute();

    loss.build();

    double t1 = MPI_Wtime();

    for (int epoch = 0; epoch < 2000; epoch ++) {
        loss.zero_grad();
        loss.run(); 
        loss.backward();
        loss.gd_step(0.1);
        loss.uniform();
        loss.combine();

        if(pid == 0) {
            cout << epoch << "\t" << loss.d[0][0] << endl;
        }
    }

    double t2 = MPI_Wtime();
    if(pid == 0) cout << "Time : " << t2 - t1 << endl;

    MPI_Finalize();

}