#include <iostream>
#include <thread>
#include <fstream>
class class_task
{
public:
void operator()() const
{
std::cout<<"Hello Concurrent World 1\n";
std::this_thread::sleep_for(std::chrono::seconds(5));
}
};
int long_run(){
	std::this_thread::sleep_for(std::chrono::seconds(10));	
	std::ofstream myfile;myfile.open ("/tmp/background.txt");
    myfile << "Writing this to a file.\n";myfile.close();
}
int main()
{
    class_task f;
    std::cout << std::thread::hardware_concurrency() << std::endl;
    std::thread t(f);std::thread t1(long_run);
    t1.detach();t.join();    
}
