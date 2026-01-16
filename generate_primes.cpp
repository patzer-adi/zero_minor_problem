#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

bool is_prime(long n)
{
    if( n<2) return false;
    if(n == 2) return true;
    if(n % 2 == 0) return false;
    for (long i = 3; i <= sqrt(n); i += 2)
    {
        if(n%i == 0) return false;
    }
    return true;
}

int main()
{
    std :: ofstream outfile("prime.txt");
    if(!outfile)
    {
        std :: cerr << "Error : could not even prime.txt for writting \n";
        return 1;
    }

    for(int i = 17; i <= 229; ++i)
    {
        if(is_prime(i))
        {
            outfile << i << "\n";
        }
    }

    outfile.close();
    std::cout <<"prime.txt generated succesfully (17 - 73)\n";
    return 0;
}
