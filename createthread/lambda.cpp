#include <thread>
#include <iostream>
using namespace std;
int main()
{ 
  //Define a lambda
  thread t1([]
  {
    cout << "Printing message with lambda" << endl;
  }
  );
  t1.join();
  
  // Define a lambda closure
  auto PrinterFunc = []() -> void { 
	  cout << "Printing message with closure lambda" << endl; 
  };
  thread t2(PrinterFunc);
  t2.join();
  return 0;
}
