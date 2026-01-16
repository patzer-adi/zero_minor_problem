#include<NTL/ZZ.h>
#include<NTL/mat_ZZ.h>
#include <iostream>

using namespace std;
using namespace NTL;

void run()
{
    long m,n;
    cout << "Enter number of rows (m):"<<endl;
    cin >> m;
    cout << "Enter number of columns(n):"<<endl;
    cin >> n;

    Mat<ZZ> arr1, arr2, add, sub, mul;
    arr1.SetDims(m,n);
    arr2.SetDims(m,n);

    cout << "Enter elements of matrix: A"<<m<<"x"<<n<<":"<<endl;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n ; j++)
        {
            cin >> arr1[i][j];
        }
    }

    cout << "Enter elements of matrix B:"<<m<<"x"<<n<<":"<<endl;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n ; j++)
        {
            ZZ val;
            cin >> val;
            arr2[i][j] = val;
        }
    }

    add = arr1 + arr2;
    cout<<"Add Matrix:"<<endl;
    cout << add <<endl;

    sub = arr1 - arr2;
    cout<<"Sub Matrix:"<<endl;
    cout << sub <<endl;

    if(m != n)
    {
        cout << "Matrix mul not possible:"<<endl;
        exit(1);
    }
    mul = arr1 * arr2;
    cout<<"Mul Matrix:"<<endl;
    cout << mul <<endl;

}

int main()
{
    run();
    return 0;
}
