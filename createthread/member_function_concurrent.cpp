#include <thread>
#include <iostream>
using namespace std;
class ReaderWriter
{
public:
  int shared_variable;
  void write_value(){
	  shared_variable = 1;
  }   
  void read_und_increase(){
	  shared_variable = shared_variable + 1;
  }
  void process(){
	  write_value();
	  std::this_thread::sleep_for(std::chrono::seconds(1));
	  read_und_increase();
	  std::this_thread::sleep_for(std::chrono::seconds(1));
	  read_und_increase();
	  std::this_thread::sleep_for(std::chrono::seconds(1));
	  read_und_increase();
	  std::this_thread::sleep_for(std::chrono::seconds(1));
	  read_und_increase();
	  toPrint();
  }
  void toPrint()
  {
    cout << "Current value:" << shared_variable << endl;
  }
  
};
int main()
{
  //Execute the toPrint() method for a 
  //specific Printer2 object 
  ReaderWriter p;
  thread t1(&ReaderWriter::process, &p);
  //Lets add more threads:
  //thread t2(&ReaderWriter::process, &p);
  //thread t3(&ReaderWriter::process, &p);
  t1.join();
  //t2.join();
  //t3.join();
  
  
  return 0;
}
