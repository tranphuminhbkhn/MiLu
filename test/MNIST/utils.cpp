#include <bits/stdc++.h>
using namespace std;


const int MAXN = 6e4 + 7;
unsigned int image[MAXN][28][28];
unsigned int num, magic, rows, cols;
unsigned int label[MAXN];

Tensor x;
Tensor y;

unsigned int in(ifstream& icin, unsigned int size) {
    unsigned int ans = 0;
    for (int i = 0; i < size; i++) {
        unsigned char x;
        icin.read((char*)&x, 1);
        unsigned int temp = x;
        ans <<= 8;
        ans += temp;
    }
    return ans;
}
void input(string s) {
    if (s == "train") {
        ifstream icin;
        icin.open("mnist_dataset/train-images-idx3-ubyte", ios::binary);
        magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);
        for (int i = 0; i < num; i++) {
            for (int x = 0; x < rows; x++) {
                for (int y = 0; y < cols; y++) {
                    image[i][x][y] = in(icin, 1);
                }
            }
        }
        icin.close();
        icin.open("mnist_dataset/train-labels-idx1-ubyte", ios::binary);
        magic = in(icin, 4), num = in(icin, 4);
        for (int i = 0; i < num; i++) {
            label[i] = in(icin, 1);
        }
        icin.close();
    }

    else if (s == "test") {
        ifstream icin;
        icin.open("mnist_dataset/t10k-images-idx3-ubyte", ios::binary);
        magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);
        for (int i = 0; i < num; i++) {
            for (int x = 0; x < rows; x++) {
                for (int y = 0; y < cols; y++) {
                    image[i][x][y] = in(icin, 1);
                }
            }
        }
        icin.close();
        icin.open("mnist_dataset/t10k-labels-idx1-ubyte", ios::binary);
        magic = in(icin, 4), num = in(icin, 4);
        for (int i = 0; i < num; i++) {
            label[i] = in(icin, 1);
        }
        icin.close();
    }
    
}


void read(int n, string s) {
    input(s);
    x = Tensor(n, 784);
    y = Tensor(n, 10);

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < 28; j++) {
            for(int k = 0; k < 28; k++) {
                x.d[i][j * 28 + k] = image[i][j][k] / 255.0;
            }
        }

        for(int j = 0; j < 10; j++) {
            y.d[i][j] = 0;
        }
        y.d[i][label[i]] = 1;
    }  

}
