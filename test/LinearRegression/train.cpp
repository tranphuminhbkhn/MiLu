#include <bits/stdc++.h>
#include "../../milu.h"
#include "mpi.h"

using namespace std;

int main() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    Tensor x = Tensor(1000, 3); x.random_init();
    Tensor y = Tensor(1000, 1);

    for(int i = 0; i < x.n; i++) {
        y.d[i][0] = 3 * x.d[i][0] + 4 * x.d[i][1] - 3 * x.d[i][2] - 1;
        y.d[i][1] = -2 * x.d[i][0] + 1 * x.d[i][1] + 4 * x.d[i][2] - 3;
    }

    Tensor z = linear(x, 3, 2);

    Tensor loss = MSELoss(y, z);

    x.distribute();
    y.distribute();

    loss.build();

    double t1 = MPI_Wtime();

    for (int epoch = 0; epoch < 2000; epoch ++) {
        loss.zero_grad();
        loss.run(); 
        loss.backward();
        loss.adam_step(1, 0.9, 0.99, epoch + 1);
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