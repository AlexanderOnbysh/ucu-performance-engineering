#include <thread>
#include <iostream>
#include <vector>

class Conto
{
    int balance;
    
    public:
    Conto() : balance(0) { }
    
    int getMoney() { 
		return balance; 
	}
	
    void addMoney(int money)
    {
       for(int i = 0; i < money; ++i)
       {
          balance++;
       }
    }
};

int testConto()
{
   Conto cntObject;
   std::vector<std::thread> threads;
   for(int i = 0; i < 5; ++i){
        threads.push_back(std::thread(&Conto::addMoney, &cntObject, 1000));
   }
 
   for(int i = 0; i < threads.size() ; i++)
   {
       threads.at(i).join();
   }
   return cntObject.getMoney();
}
 
int main()
{
 
  int val = 0;
  for(int k = 0; k < 1000; k++)
  {
     if((val = testConto()) != 5000)
     {
       std::cout << "Error at balance, "<< k ;
       std::cout << " Balance at conto:" << val << std::endl;
     }
  }
  return 0;
}
