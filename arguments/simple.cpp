#include <iostream>
#include <string>
#include <thread>
void threadFunction(int x, std::string str)
{
    std::cout<<"Integer = "<< x   <<std::endl;
    std::cout<<"String = " << str <<std::endl;
}
int main()  
{
    int x = 12;
    std::string str = "Lorem ipsum";
    std::thread threadObj(threadFunction, x, str);
    threadObj.join();
    return 0;
}
