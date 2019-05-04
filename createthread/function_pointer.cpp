#include <thread>
#include <iostream>
using namespace std;
void PrintThread()
{
  cout << "Printing message in child thread by function pointer" << endl;
}
int main()
{
  //Creating a thread with a function pointer/declared function name
  thread t1(PrintThread);
  t1.join();
  
  return 0;
}
