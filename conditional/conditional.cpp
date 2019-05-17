#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

class DataProcessor
{
std::mutex mymutex;
bool DataLoaded;
std::condition_variable mycondVar;
public:
DataProcessor(){DataLoaded = false;}
void loadData()
{   
   std::cout<<"Loading Data from slow WEB resource"<<std::endl;
   std::this_thread::sleep_for(std::chrono::milliseconds(1000));
   std::lock_guard<std::mutex> guard(mymutex);
   DataLoaded = true;
   mycondVar.notify_one();
}
void Task()
{
    std::cout<<"Do Work while data loaded"<<std::endl;
    std::unique_lock<std::mutex> mlock(mymutex);
    mycondVar.wait(mlock, std::bind(&DataProcessor::isDataLoaded, this));
    std::cout<<"Work with loaded Data"<<std::endl;
}
bool isDataLoaded()
{
return DataLoaded;
}
};
int main()
{
DataProcessor app;
std::thread thread_1(&DataProcessor::Task, &app);
std::thread thread_2(&DataProcessor::loadData, &app);
thread_2.join();
thread_1.join();
return 0;
}
