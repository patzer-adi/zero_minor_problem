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

    Mat<ZZ> arr;
    arr.SetDims(m,n);

    cout << "Enter elements of matrix:"<<m<<"x"<<n<<":"<<endl;
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n ; j++)
        {
            ZZ val;
            cin >> val;
            arr[i][j] = val;
        }
    }

    cout<<"Matrix:"<<endl;
    cout << arr <<endl;


}

int main()
{
    run();
    return 0;
}

