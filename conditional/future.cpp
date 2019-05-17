#include <iostream>
#include <thread>
#include <future>
 
void mythread(std::promise<int> * promObj)
{
    std::cout << "Inside mythread" << std::endl;
    promObj->set_value(17);
}
 
int main()
{
    std::promise<int> promiseObj;
    std::future<int> futureObj = promiseObj.get_future();
    std::thread t(mythread, &promiseObj);
    std::cout << futureObj.get() << std::endl;
    t.join();
    return 0;
}
