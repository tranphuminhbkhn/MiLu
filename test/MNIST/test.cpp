#include <bits/stdc++.h>
#include "../../milu.h"
#include "mpi.h"
#include "utils.cpp"

using namespace std;

int main() {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int ntest = 10000;
    read(ntest, "test");

    Tensor h = relu(linear(x, 784, 50));
    Tensor out = linear(h, 50, 10);
    Tensor z = softmax(out);

    Tensor loss = CrossEntropyLoss(y, z);

    x.distribute();

    loss.load_graph("model.txt");
    z.build();

    double t1 = MPI_Wtime();
    z.run();
    z.combine();
    int r = 0;
    for (int i = 0; i < ntest; i++) {
        int a = 0;
        for(int j = 1; j < 10; j++) {
            if (z.d[i][j] > z.d[i][a]) {
                a = j;
            }
        }
        int b = 0;
        for(int j = 1; j < 10; j++) {
            if (y.d[i][j] > y.d[i][b]) {
                b = j;
            }
        }
        if (a == b) r ++;
    }
    if (pid == 0) cout << "Acc : " << double(r) / ntest << endl;
    double t2 = MPI_Wtime();
    if(pid == 0) cout << "Time : " << t2 - t1 << endl;
    MPI_Finalize();
    return 0;
}