#include <bits/stdc++.h>
#include "../../milu.h"
#include "mpi.h"
#include "utils.cpp"

using namespace std;

int main() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    read(6000, "train");

    Tensor h = relu(linear(x, 784, 50));
    h.dropout(0.2);
    Tensor out = linear(h, 50, 10);
    Tensor z = softmax(out);

    Tensor loss = CrossEntropyLoss(y, z);

    x.distribute();
    y.distribute();
    // loss.load_graph("model.txt");
    loss.build();

    double t1 = MPI_Wtime();

    for (int epoch = 0; epoch < 200; epoch ++) {
        loss.zero_grad();
        loss.run(); 
        loss.backward();
        loss.adam_step(0.01, 0.9, 0.99, epoch + 1);
        loss.uniform();
        loss.combine();

        if(pid == 0) {
            cout << epoch << "\t" << loss.d[0][0] << endl;
        }
    }

    double t2 = MPI_Wtime();
    if(pid == 0) cout << "Time : " << t2 - t1 << endl;

    loss.save_graph("model.txt");

    MPI_Finalize();
    return 0;
}