#include<NTL/ZZ.h>

using namespace std;
using namespace NTL;

void run()
{
    ZZ n1,n2,ans;

    cout<<"Enter a number n1:\n"<<endl;
    cin >> n1;
    cout<<"Enter a number n2:\n"<<endl;
    cin >> n2;

    ans = n1 + n2;

    cout <<"Output n1 + n2: "<<ans<<endl;

    ans = n1 * n2;

    cout <<"Output n1 * n2: "<<ans<<endl;

    ans = n1 - n2;

    cout << "Output n1 - n2: "<<ans<<endl;

    ans = n1 / n2;

    cout << "Output n1 / n2: "<<ans<<endl;

    ans = n1 % n2;

    cout << "Output n1 % n2: "<< ans<<endl;

}


int main()
{
    run();
    return 0;
}
