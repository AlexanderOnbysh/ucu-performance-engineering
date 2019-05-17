#include<iostream>
#include<thread>
#include<mutex>

class DataProcessor
{
std::mutex mymutex;
bool DataLoaded;
public:
DataProcessor(){DataLoaded = false;}
void loadData()
{   
   std::cout<<"Loading Data from slow WEB resource"<<std::endl;
   std::this_thread::sleep_for(std::chrono::milliseconds(1000));
   std::lock_guard<std::mutex> guard(mymutex);
   DataLoaded = true;
}
void Task()
{
    std::cout<<"Do Work while data loaded"<<std::endl;
    mymutex.lock();
    while(DataLoaded != true)
    {
        mymutex.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        mymutex.lock();
    }
    mymutex.unlock();
    std::cout<<"Work with loaded Data"<<std::endl;
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
