#include <iostream>
#include <thread>
void thread_function()
{
    std::cout<<"Inside T. ID"<<std::this_thread::get_id()<<std::endl;    
}
int main()  
{
    std::thread t1(thread_function);
    std::thread t2(thread_function);
 
    std::cout<<"From Main ID T1 = "<<t1.get_id()<<std::endl;    
    std::cout<<"From Main ID T2 = "<<t2.get_id()<<std::endl;    
 
    t1.join();    
    t2.join();    
    return 0;
}
