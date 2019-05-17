#include <iostream>
#include <thread>
#include <future>
#include <vector>
#include <numeric>

template <typename Iterator>
typename std::iterator_traits<Iterator>::value_type
adder(Iterator begin, Iterator end)
{
    using T = typename std::iterator_traits<Iterator>::value_type;
    return std::accumulate(begin, end, T());
}

template <typename Iterator>
typename std::iterator_traits<Iterator>::value_type
parallel_sum(Iterator begin, Iterator end)
{
    using T = typename std::iterator_traits<Iterator>::value_type;
    auto midpoint = begin + std::distance(begin, end) / 2;
    std::future<T> sum1 = std::async(std::launch::async, adder<Iterator>, begin, midpoint);
    std::future<T> sum2 = std::async(std::launch::async, adder<Iterator>, midpoint, end);
    return sum1.get() + sum2.get();
}

int main()
{
    std::vector<int> v;
    for(int i = 0; i < 1024*1024*1024; ++i) {
        v.push_back(1);
    }
    int total = parallel_sum(v.begin(), v.end());
    std::cout << total << "\n";
}
