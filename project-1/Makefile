run: app
	./app

run-optimized: app-optimized
	./app

app-optimized: main.cpp
	gcc  -O3 -ftree-vectorize -g3 -fno-omit-frame-pointer -lstdc++ -o app main.cpp

app : main.cpp
	g++ -o app main.cpp

clean:
	rm app