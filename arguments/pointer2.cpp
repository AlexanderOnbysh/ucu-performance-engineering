#include <iostream>
#include <thread>
void threadFunction(int * q)
{
    std::chrono::milliseconds t( 1000 );
    std::this_thread::sleep_for( t );
    std::cout<<"In Thread :  "" : q = "<< *q << " " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() << std::endl;
    std::this_thread::sleep_for( t );
    *q = 113;
    std::cout<<"In Thread :  "" : q = "<< *q  << " " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() << std::endl;
}
void startThread()
{
    int x = 10;
    std::cout<<"Main Thread :  "" : x = "<< x << " " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() << std::endl;
    std::thread t(threadFunction,&x);
    t.detach();
    std::cout<<"Main Thread :  "" : x = "<< x << " " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() << std::endl;

    // std::chrono::milliseconds test( 2000 );
    // std::this_thread::sleep_for( test );
}
int main()
{
    startThread();
    std::chrono::milliseconds t( 2000 );
    std::this_thread::sleep_for( t );
    return 0;
}
