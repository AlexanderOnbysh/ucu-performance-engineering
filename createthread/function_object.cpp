#include <thread>
#include <iostream>
using namespace std;
//Create the function object
class Printer
{
public:
  void operator() () const
  {
     cout << "Print message from the function object" << endl;
  }
};
int main()
{
  //Creating a thread with an function object
  Printer p1;
  thread t1(p1);
  t1.join();
  return 0;
}
