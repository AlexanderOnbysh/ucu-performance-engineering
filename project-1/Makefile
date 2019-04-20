run: clean app
	./app

run-optimized: clean clean app-optimized
	./app

app-optimized: main.cpp
	gcc -O3 -march=native -g3 -fno-omit-frame-pointer -lstdc++ -o app main.cpp

app : main.cpp
	g++ -O0 -o app main.cpp

clean:
	rm -f app