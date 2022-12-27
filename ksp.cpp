#include <iostream>
#include <math.h>

using namespace std;

int main()
{
    float g = 9.81;
    int length = 1000;

    float v = 0;
    int Impulse = 300;
    int M_start = 2270000;
    int M_end = 860000;
    int step = (M_start - M_end) / length;

    for (int i = 0; i < length; i ++)
    {
        v += Impulse * g * log(1.0 * (M_start - i * step)/ (M_start - (i + 1) * step));
        cout << v << "\t";
        if (i % 10 == 9)
        {
            cout.precision(9);
            cout << endl;
        }   
    }
}