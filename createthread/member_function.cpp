#include <thread>
#include <iostream>
using namespace std;
class Printer2
{
public:
  void toPrint()
  {
    cout << "Print with method" << endl;
  }
};
int main()
{
  //Execute the toPrint() method for a 
  //specific Printer2 object 
  Printer2 p;
  thread t1(&Printer2::toPrint, &p);
  t1.join();
  return 0;
}
