run: clean app
	./app

run-optimized: clean clean app-optimized
	./app

app-optimized: main.cpp
	gcc -O3 -march=native -g3 -fno-omit-frame-pointer -lstdc++ -o app main.cpp -I /opt/OpenBLAS/include/ -L /opt/OpenBLAS/lib -lopenblas -lpthread

app : main.cpp
	g++ -O0 -o app main.cpp -I /opt/OpenBLAS/include/ -L /opt/OpenBLAS/lib -lopenblas -lpthread

clean:
	rm -f app