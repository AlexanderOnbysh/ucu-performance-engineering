#include <iostream>
#include <thread>
void threadFunction(int const & x)
{
    int & p = const_cast<int &>(x);
    p++;
    std::cout<<"Thread x = "<< x <<std::endl;
}
int main()
{
    int x = 10;
    std::cout<<"In Main Start x = "<<x<<std::endl;
    std::thread t(threadFunction, std::ref(x));
    t.join();
    std::cout<<"In Main End x = "<<x<<std::endl;
    return 0;
}
